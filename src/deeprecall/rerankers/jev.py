"""TypeSafe jev reranker: a calibrated decision model answering one yes/no question per passage.

It can also classify a question into one of KINDS with a single choice request, and score some kinds with
their own prompt (KIND_PROMPTS): recall uses both when `[recall] question_kinds = true`.

Needs an API key in $TYPESAFE_API_KEY (or `api_key_env` in config). Pricing is per input token;
set usd_per_mtok to your plan's price (0.042 at the time of writing, https://docs.typesafe.ai/models).
"""
from __future__ import annotations

import copy
import os
import socket
from urllib.parse import urlparse

from ._http import pmap, post_json
from .base import Passage, Reranker, RerankerError

PROMPT = ('Does `content` explicitly state the specific fact that answers this question: "{q}"? '
          'Yes only if the answer itself is written in the text, not merely the same topic.')

KINDS = {
    "single_fact": "Asks for one fact the user or assistant stated once.",
    "latest_value": "Asks for the current value of something that may have changed over time.",
    "aggregate": "Asks for a count, total or list that combines facts from several separate conversations.",
    "temporal": "Asks when something happened, the order of events, or how much time passed; the answer depends on dates.",
    "preference": "Asks for a recommendation, suggestion or advice where the user's own tastes and circumstances matter.",
}
# A recommendation's answer is advice, not a fact, so "states the answer" asks the wrong thing of the user's
# remarks that should shape it.
KIND_PROMPTS = {
    "preference": ('Does `content` reveal a preference, interest, habit or circumstance of the user that a good '
                   'response to this request should take into account: "{q}"? Yes if the user says something about '
                   'themselves that should shape the response. No if it only discusses the same topic.'),
}


def _key_from_file(path: str | None, var: str) -> str:
    """Read a key from a file holding either the bare key or VAR=key lines."""
    if not path:
        return ""
    try:
        text = open(os.path.expanduser(path)).read()
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip().strip('"\'')
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    return lines[0] if len(lines) == 1 and "=" not in lines[0] else ""


def _reachable(url: str, timeout: float = 1.5) -> bool:
    u = urlparse(url)
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=timeout).close()
        return True
    except OSError:
        return False


class JevReranker(Reranker):
    name = "jev"
    window_words = 600
    widen_at = 0.7

    def __init__(self, api_key_env: str = "TYPESAFE_API_KEY", api_key_file: str | None = None,
                 model: str = "jev-latest",
                 base_url: str = "https://api.typesafe.ai/v1", usd_per_mtok: float = 0.042,
                 workers: int = 32, prompt: str = PROMPT, widen_at: float | None = None,
                 window_words: int | None = None, kind_prompts: dict | None = None,
                 fallback_base_url: str | None = None, fallback_model: str = "jev-latest",
                 fallback_usd_per_mtok: float = 0.042, fallback_workers: int = 32,
                 primary_timeout: float = 15, **_):
        """With fallback_base_url set, base_url is a local jev-compatible server (e.g. a tower running
        clef-flash). If it is unreachable or errors at the transport level, this reranker (and every
        for_kind() copy, which share _route) switches to the fallback for the rest of the process."""
        self.key = os.environ.get(api_key_env, "") or _key_from_file(api_key_file, api_key_env)
        if not self.key and not fallback_base_url:
            raise RerankerError(f"jev: set ${api_key_env} or api_key_file")
        self.prompt = prompt
        # 2026-10-04: a busy tower (bulk job queued ahead) held one recall for 359s at timeout=60 x retries;
        # one 15s try on the primary, then sticky fallback.
        self.primary_timeout = float(primary_timeout)
        self._route = {"url": base_url.rstrip("/") + "/systemone", "model": model,
                       "usd_per_token": usd_per_mtok / 1e6, "workers": workers, "fell_back": None}
        self._fallback = None
        if fallback_base_url:
            self._fallback = {"url": fallback_base_url.rstrip("/") + "/systemone", "model": fallback_model,
                              "usd_per_token": fallback_usd_per_mtok / 1e6, "workers": fallback_workers}
            if not _reachable(base_url):
                self._fall_back(f"{base_url} unreachable")
        self.kind_prompts = KIND_PROMPTS if kind_prompts is None else dict(kind_prompts)
        if widen_at is not None:
            self.widen_at = float(widen_at)
        if window_words:
            self.window_words = int(window_words)

    # route fields are read through properties so a fallback switch is seen by every copy
    model = property(lambda self: self._route["model"])
    url = property(lambda self: self._route["url"])
    workers = property(lambda self: self._route["workers"])
    usd_per_token = property(lambda self: self._route["usd_per_token"])

    def _fall_back(self, why: str) -> bool:
        if not self._fallback or self._route["fell_back"]:
            return False
        if not self.key:
            raise RerankerError(f"jev: primary failed ({why}) and no key for fallback")
        self._route.update(self._fallback, fell_back=why)
        return True

    def _ask(self, state: dict, questions: dict) -> dict:
        # Tower-only (no fallback, Mike 2026-10-04 "only use clef flash") is also a primary route: short
        # timeout, 2 tries, then RerankerError -> recall returns first-stage results instead of hanging.
        local = not self._route["fell_back"] and (self._fallback is not None or self.usd_per_token == 0)
        primary = local and self._fallback is not None
        try:
            return post_json(self.url, {"state": state, "model": self.model, "questions": questions},
                             {"Authorization": "Bearer " + self.key} if self.key else {},
                             timeout=self.primary_timeout if local else 60,
                             retries=(1 if primary else 2) if local else 4)
        except RerankerError as e:
            transport = not str(e).startswith("HTTP 4")
            if primary and transport and self._fall_back(str(e)[:120]):
                return self._ask(state, questions)
            raise

    def classify(self, question: str) -> str | None:
        r = self._ask({"question": question},
                      {"kind": {"type": "choice", "instructions": "What kind of memory lookup does `question` need?",
                                "criteria": KINDS}})
        self.last_tokens = int(r.get("usage", {}).get("input_tokens", 0))
        kind = (r.get("answers", {}).get("kind") or {}).get("choice")
        return kind if kind in KINDS else None

    def for_kind(self, kind: str | None) -> "JevReranker":
        if kind not in self.kind_prompts:
            return self
        tuned = copy.copy(self)
        tuned.prompt = self.kind_prompts[kind]
        return tuned

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        qs = {"q": {"type": "noul", "instructions": self.prompt.format(q=question)}}
        toks = []

        def one(p: Passage) -> float:
            text = p.text
            for _ in range(3):
                try:
                    r = self._ask({"path": p.path, "content": text}, qs)
                    toks.append(int(r.get("usage", {}).get("input_tokens", len(text) // 3)))
                    return float(r["answers"]["q"]["noul"])
                except RerankerError as e:
                    if "max_tokens" in str(e) and len(text) > 4000:
                        text = text[: len(text) // 2]
                        continue
                    raise
            return 0.0

        out = pmap(one, passages, self.workers)
        self.last_tokens = sum(toks)
        return out
