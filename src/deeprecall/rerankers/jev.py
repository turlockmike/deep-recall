"""TypeSafe jev reranker: a calibrated decision model answering one yes/no question per passage.

Needs an API key in $TYPESAFE_API_KEY (or `api_key_env` in config). Pricing is per input token;
set usd_per_mtok to your plan's price (0.042 at the time of writing, https://docs.typesafe.ai/models).
"""
from __future__ import annotations

import os

from ._http import pmap, post_json
from .base import Passage, Reranker, RerankerError

PROMPT = ('Does `content` explicitly state the specific fact that answers this question: "{q}"? '
          'Yes only if the answer itself is written in the text, not merely the same topic.')


class JevReranker(Reranker):
    name = "jev"
    window_words = 600
    widen_at = 0.7

    def __init__(self, api_key_env: str = "TYPESAFE_API_KEY", model: str = "jev-latest",
                 base_url: str = "https://api.typesafe.ai/v1", usd_per_mtok: float = 0.042,
                 workers: int = 32, prompt: str = PROMPT, widen_at: float | None = None,
                 window_words: int | None = None, **_):
        self.key = os.environ.get(api_key_env, "")
        if not self.key:
            raise RerankerError(f"jev: set ${api_key_env}")
        self.model, self.url, self.workers, self.prompt = model, base_url.rstrip("/") + "/systemone", workers, prompt
        self.usd_per_token = usd_per_mtok / 1e6
        if widen_at is not None:
            self.widen_at = float(widen_at)
        if window_words:
            self.window_words = int(window_words)

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        qs = {"q": {"type": "noul", "instructions": self.prompt.format(q=question)}}
        toks = []

        def one(p: Passage) -> float:
            text = p.text
            for _ in range(3):
                try:
                    r = post_json(self.url, {"state": {"path": p.path, "content": text}, "model": self.model,
                                             "questions": qs}, {"Authorization": "Bearer " + self.key}, timeout=60)
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
