"""Answer-aware recall.

  1. pool   = hybrid search top-k  ∪  rare-term leg
  2. every candidate file -> heading-bounded windows (reranker's preferred size, <= max_windows/file)
  3. reranker scores every window: P(window states the answer); file score = best window
  4. rank; if best < widen_at, widen once to hybrid top-widen_k and score only the new files
With question_kinds, the reranker first classifies the question (one small request). The kind picks the
reranker's prompt, and a kind in widen_kinds widens whatever the best score: a count or a timeline needs
every mention, and one confident hit says nothing about the rest.
Non-question queries (no '?', no leading question word) skip step 2-4: first-stage order is
already right for keyword lookups, and "does this state the answer?" is ill-posed for them.
Any reranker failure (budget, network, auth) falls back to first-stage order.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from . import index as idx
from .budget import Budget
from .config import Config
from .markdown import read_doc, split_frontmatter, windows
from .rare import RareIndex
from .rerankers.base import Passage, Reranker, load
from .search import search

QWORDS = {"what", "which", "when", "where", "why", "how", "who", "whom", "whose", "is", "are", "was", "were",
          "does", "do", "did", "can", "could", "should", "would", "will", "has", "have", "had", "list", "find"}


STOPWORDS = set("""a an the of to in on for and or but is are was were be been do does did what which when
where why how who whom whose with from by at as that this it its we our i my you your they their he she
have has had not no can could should would will there than then into about after before over under""".split())


def _words(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9_-]*", t.lower()) if w not in STOPWORDS and len(w) > 1]


def prefilter(question: str, wins: list, per_file: int) -> list:
    """Keep each file's `per_file` windows with the most (idf-weighted) question words."""
    qw = set(_words(question))
    if not qw or per_file <= 0:
        return wins
    df = Counter(w for win in wins for w in qw & set(_words(win.text)))
    n = len(wins) or 1
    def score(win):
        ws = set(_words(win.text))
        return sum(math.log(1 + n / df[w]) for w in qw & ws)
    by_file: dict[str, list] = {}
    for w in wins:
        by_file.setdefault(w.path, []).append(w)
    out = []
    for path, ws in by_file.items():
        out += sorted(ws, key=score, reverse=True)[:per_file] if len(ws) > per_file else ws
    return out


def is_question(q: str) -> bool:
    w = q.strip().split()
    return q.strip().endswith("?") or (len(w) >= 4 and w[0].lower() in QWORDS)


@dataclass
class Hit:
    path: str
    score: float | None
    section: str = ""
    passage: str = ""


@dataclass
class Result:
    query: str
    hits: list[Hit]
    mode: str
    pool: int
    widened: bool = False
    usd: float = 0.0
    tokens: int = 0
    secs: float = 0.0
    note: str = ""
    kind: str | None = None
    extra: dict = field(default_factory=dict)


class Recaller:
    def __init__(self, cfg: Config, reranker: Reranker | None = None):
        self.cfg = cfg
        self._reranker = reranker
        self.caller = "live"  # eval.run sets "eval": its spend is capped separately (budget.py CALLER SPLIT)
        self._rare = None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = load(self.cfg.reranker)
        return self._reranker

    def rare(self) -> RareIndex:
        if self._rare is None:
            self._rare = RareIndex([(d, a) for d, a, _r in idx.iter_files(self.cfg)], self.cfg.roots)
        return self._rare

    def first_stage(self, q: str, k: int) -> list[str]:
        return [p for p, _ in search(self.cfg, q, k)]

    def _classify(self, q: str, budget: Budget) -> tuple[str | None, int]:
        """The question's kind and the tokens it cost; a failed classification loses only the kind, and
        whatever it was billed is still recorded."""
        rr, kind = None, None
        try:
            rr = self.reranker
            rr.last_tokens = 0
            budget.check((len(q) + 900) / 3 * rr.usd_per_token)
            kind = rr.classify(q)
        except Exception:
            kind = None
        tokens = rr.last_tokens if rr is not None else 0
        if tokens:
            budget.record(rr.name, tokens, tokens * rr.usd_per_token)
        return kind, tokens

    def _score_files(self, q, paths, cache, passages, budget, rr=None):
        rr = rr or self.reranker
        todo = [p for p in paths if p not in cache]
        wins: list[Passage] = []
        for p in todo:
            try:
                _fm, body = split_frontmatter(read_doc(idx.resolve(self.cfg, p)))
            except OSError:
                cache[p] = (0.0, "")
                continue
            wins += [Passage(p, h, t) for h, t in windows(body, rr.window_words, self.cfg.max_windows)]
        if rr.prefilter_per_file:
            wins = prefilter(q, wins, rr.prefilter_per_file)
        for p in todo:
            cache.setdefault(p, (0.0, ""))
        if not wins:        # nothing new to score; last_tokens still holds the previous call's count
            return 0
        est = rr.estimate_tokens(q, wins) * rr.usd_per_token
        budget.check(est)
        rr.last_tokens = 0
        scores = rr.score(q, wins)
        tokens = rr.last_tokens or (rr.estimate_tokens(q, wins) if rr.usd_per_token else 0)
        budget.record(rr.name, tokens, tokens * rr.usd_per_token)
        for w, s in zip(wins, scores):
            if s > cache[w.path][0]:
                cache[w.path] = (s, w.heading)
                passages[w.path] = w.text
        return tokens

    def recall(self, q: str, top: int = 5, k: int | None = None, rerank: bool = True) -> Result:
        t0, cfg = time.time(), self.cfg
        k = k or cfg.k
        fs = self.first_stage(q, k)
        pool = list(dict.fromkeys(fs + self.rare().leg(q, cfg.rare_max_df)))
        if not rerank or (cfg.question_gate and not is_question(q)) or cfg.reranker.get("backend") == "none":
            mode = "keyword" if rerank and cfg.question_gate and not is_question(q) else "first-stage"
            res = Result(q, [Hit(p, None) for p in pool[:top]], mode, len(pool), secs=round(time.time() - t0, 2))
            self._log(res)
            return res
        budget = Budget(cfg.ledger, cfg.max_usd_per_query,
                        cfg.eval_daily_cap_usd if self.caller == "eval" else cfg.daily_cap_usd, self.caller)
        cache: dict[str, tuple[float, str]] = {}
        passages: dict[str, str] = {}
        res = Result(q, [], "rerank:" + str(cfg.reranker.get("backend")), len(pool))
        res.extra["first_stage_top3"] = fs[:3]
        try:
            if cfg.question_kinds:
                res.kind, tokens = self._classify(q, budget)
                res.tokens += tokens
            rr = self.reranker.for_kind(res.kind)
            res.tokens += self._score_files(q, pool, cache, passages, budget, rr)
            ranked = sorted(pool, key=lambda p: (-cache[p][0], pool.index(p)))
            widen_at = rr.widen_at if cfg.widen_at == "auto" else float(cfg.widen_at)
            unsure = ranked and cache[ranked[0]][0] < widen_at
            if ranked and (unsure or res.kind in cfg.widen_kinds) and cfg.widen_k > k:
                wide = list(dict.fromkeys(pool + self.first_stage(q, cfg.widen_k)))
                if len(wide) > len(pool):
                    try:
                        res.tokens += self._score_files(q, wide, cache, passages, budget, rr)
                        ranked = sorted(wide, key=lambda p: (-cache[p][0], wide.index(p)))
                        res.widened, res.pool = True, len(wide)
                    except Exception as e:  # keep the first pass result
                        res.note = f"widen skipped: {e}"
            res.hits = [Hit(p, cache[p][0], cache[p][1], passages.get(p, "") if i == 0 else "")
                        for i, p in enumerate(ranked[:top])]
            res.extra["widen_at"] = widen_at
        except Exception as e:
            res.mode = "first-stage (reranker unavailable)"
            res.note = f"{type(e).__name__}: {str(e)[:200]}"
            res.hits = [Hit(p, None) for p in pool[:top]]
        res.usd = round(budget.query_usd, 6)
        res.secs = round(time.time() - t0, 2)
        self._log(res)
        return res

    def _log(self, r: Result) -> None:
        try:
            self.cfg.recall_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.recall_log, "a") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "q": r.query, "mode": r.mode,
                                    "top": [[h.path, h.score] for h in r.hits[:3]], "pool": r.pool,
                                    "widened": r.widened, "usd": r.usd, "secs": r.secs, "note": r.note, "kind": r.kind,
                                    "first_stage_top3": r.extra.get("first_stage_top3"),
                                    "widen_at": r.extra.get("widen_at")}) + "\n")
        except OSError:
            pass
