"""TypeSafe jev reranker: a calibrated decision model answering one yes/no question per passage.

It can also classify a question into one of KINDS with a single choice request, and score some kinds with
their own prompt (KIND_PROMPTS): recall uses both when `[recall] question_kinds = true`.

Needs an API key in $TYPESAFE_API_KEY (or `api_key_env` in config). Pricing is per input token;
set usd_per_mtok to your plan's price (0.042 at the time of writing, https://docs.typesafe.ai/models).
"""
from __future__ import annotations

import copy
import os

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


class JevReranker(Reranker):
    name = "jev"
    window_words = 600
    widen_at = 0.7

    def __init__(self, api_key_env: str = "TYPESAFE_API_KEY", api_key_file: str | None = None,
                 model: str = "jev-latest",
                 base_url: str = "https://api.typesafe.ai/v1", usd_per_mtok: float = 0.042,
                 workers: int = 32, prompt: str = PROMPT, widen_at: float | None = None,
                 window_words: int | None = None, kind_prompts: dict | None = None, **_):
        self.key = os.environ.get(api_key_env, "") or _key_from_file(api_key_file, api_key_env)
        if not self.key:
            raise RerankerError(f"jev: set ${api_key_env} or api_key_file")
        self.model, self.url, self.workers, self.prompt = model, base_url.rstrip("/") + "/systemone", workers, prompt
        self.usd_per_token = usd_per_mtok / 1e6
        self.kind_prompts = KIND_PROMPTS if kind_prompts is None else dict(kind_prompts)
        if widen_at is not None:
            self.widen_at = float(widen_at)
        if window_words:
            self.window_words = int(window_words)

    def _ask(self, state: dict, questions: dict) -> dict:
        return post_json(self.url, {"state": state, "model": self.model, "questions": questions},
                         {"Authorization": "Bearer " + self.key}, timeout=60)

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
