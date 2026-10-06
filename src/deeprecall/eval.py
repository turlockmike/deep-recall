"""Small evaluation harness: first-stage vs recall on a labeled question file.

Question file: JSON list or JSONL of {"q": "...", "gold": ["relative/path.md", ...]} where gold
lists every file that states the answer. Exact-path scoring undercounts when other files state
the same fact in other words; judge those by hand (or with a blind LLM judge) for headline numbers.
"""
from __future__ import annotations

import json
import sys

from .config import load
from .recall import Recaller


def _read(path: str) -> list[dict]:
    txt = open(path).read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def _experiment_ban() -> str | None:
    """Mike's jev-experiment ban (2026-09-29, ~/.config/jev/NO-EXPERIMENTS). An eval is an experiment that
    spends TypeSafe credits through deep-recall's OWN client, so the gate lives here, at the spend path,
    not only in the bash wrapper (auditor D427-V2-1). Fail closed: if jevlib can't be imported, refuse."""
    import os
    flag = os.path.expanduser("~/.config/jev/NO-EXPERIMENTS")
    if not os.path.exists(flag):
        return None
    try:
        sys.path.insert(0, os.path.expanduser("~/.local/lib/python"))
        import jevlib
        if jevlib.is_interactive():
            return None
    except Exception:
        pass
    return f"deeprecall eval REFUSED: jev experiments are banned outside a live session with Mike ({flag})"


def first_stage(paths: list[str], limit: int | None, top: int, as_json: bool, depth: int = 20) -> int:
    """First-stage-only scoring (hybrid BM25 + vector, no reranker, no LLM, $0): hit@1 / hit@top / hit@depth / MRR@depth
    and per-query latency. Questions whose gold files are not in the index are skipped (reported as `skipped`).
    This is the embedder A/B instrument: run it twice with two DEEPRECALL_CONFIGs (e.g. bge vs EmbeddingGemma 2 index)."""
    import time
    from .config import load as _load
    from .index import connect
    from .search import search
    cfg = _load()
    db = connect(cfg)
    indexed = {r[0] for r in db.execute("SELECT path FROM docs")}
    db.close()
    qs = [d for p in paths for d in _read(p)]
    qs = qs[:limit] if limit else qs
    rows, skipped = [], 0
    for d in qs:
        gold = [p for p in (d.get("gold") or d.get("ev") or []) if p in indexed]
        if not gold:
            skipped += 1
            continue
        t0 = time.time()
        got = [p for p, _ in search(cfg, d["q"], depth)]
        secs = time.time() - t0
        rank = next((i + 1 for i, p in enumerate(got) if p in gold), 0)
        rows.append({"q": d["q"], "rank": rank, "secs": round(secs, 3), "gold": gold, "top": got[:3]})
    n = len(rows)
    lat = sorted(r["secs"] for r in rows)
    out = {"n": n, "skipped": skipped, "index": str(cfg.index), "embedding": {"model": cfg.embed_model, **{k: v for k, v in cfg.embedding.items() if k != "model"}},
           "hit@1": sum(r["rank"] == 1 for r in rows), f"hit@{top}": sum(0 < r["rank"] <= top for r in rows),
           f"hit@{depth}": sum(r["rank"] > 0 for r in rows), f"mrr@{depth}": round(sum(1 / r["rank"] for r in rows if r["rank"]) / max(n, 1), 4),
           "secs_median": lat[n // 2] if n else None, "secs_p90": lat[int(n * 0.9)] if n else None}
    print(json.dumps({**out, "rows": rows} if as_json else out, indent=1))
    return 0


def run(path: str, limit: int | None, top: int, max_usd: float, as_json: bool) -> int:
    ban = _experiment_ban()
    if ban:
        print(ban, file=sys.stderr)
        return 3
    qs = _read(path)[:limit] if limit else _read(path)
    cfg = load()
    if not cfg.eval_daily_cap_usd:   # Mike 2026-10-01 20:03: jev = deeprecall live rerank only; no evals
        print("deeprecall eval REFUSED: eval_daily_cap_usd = 0 (Mike 2026-10-01: jev for live rerank only, no jev evals)", file=sys.stderr)
        return 3
    rc = Recaller(cfg)
    rc.caller = "eval"  # own rolling cap; never charged against live recall
    rows, spent = [], 0.0
    for i, d in enumerate(qs, 1):
        gold = set(d["gold"])
        base = [p for p in rc.first_stage(d["q"], max(top, 10))]
        r = rc.recall(d["q"], top=top)
        spent += r.usd
        got = [h.path for h in r.hits]
        rows.append({"q": d["q"], "base1": bool(base[:1]) and base[0] in gold, "base3": bool(gold & set(base[:top])),
                     "rec1": bool(got[:1]) and got[0] in gold, "rec3": bool(gold & set(got[:top])),
                     "mode": r.mode, "usd": r.usd, "secs": r.secs})
        b1 = sum(x["base1"] for x in rows); r1 = sum(x["rec1"] for x in rows)
        print(f"[{i}/{len(qs)}] first-stage@1 {b1}  recall@1 {r1}  spent ${spent:.4f}", file=sys.stderr, flush=True)
        if spent >= max_usd:
            print(f"stopping: spend ${spent:.4f} reached --max-usd {max_usd}", file=sys.stderr)
            break
    n = len(rows)
    s = {k: sum(x[k] for x in rows) for k in ("base1", "base3", "rec1", "rec3")}
    out = {"n": n, "first_stage": {"hit@1": s["base1"], f"hit@{top}": s["base3"]},
           "recall": {"hit@1": s["rec1"], f"hit@{top}": s["rec3"]}, "usd": round(spent, 4),
           "mean_secs": round(sum(x["secs"] for x in rows) / max(n, 1), 2)}
    print(json.dumps(out if not as_json else {**out, "rows": rows}, indent=1))
    return 0
