"""Any OpenAI-compatible chat endpoint as a yes/no answerability judge, using token logprobs.

Works with OpenAI, OpenRouter, Together, Groq, vLLM, llama.cpp server, LM Studio, Ollama's
OpenAI endpoint... anything that returns `logprobs` for chat completions. The score is
P(yes) = exp(lp_yes) / (exp(lp_yes) + exp(lp_no)) over the first generated token.

Config:
    [reranker]
    backend = "openai"
    base_url = "https://api.openai.com/v1"
    model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"
    usd_per_mtok = 0.15
"""
from __future__ import annotations

import math
import os

from ._http import pmap, post_json
from .base import Passage, Reranker, RerankerError

SYSTEM = ("You judge whether a passage states the answer to a question. Reply with exactly one word: "
          "Yes if the passage explicitly states the specific fact that answers the question, otherwise No.")


class OpenAIYesNoReranker(Reranker):
    name = "openai"
    window_words = 600
    widen_at = 0.7

    def __init__(self, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini",
                 api_key_env: str = "OPENAI_API_KEY", usd_per_mtok: float = 0.0, workers: int = 8,
                 widen_at: float | None = None, window_words: int | None = None, **_):
        self.key = os.environ.get(api_key_env, "")
        self.url, self.model, self.workers = base_url.rstrip("/") + "/chat/completions", model, workers
        self.usd_per_token = usd_per_mtok / 1e6
        if widen_at is not None:
            self.widen_at = float(widen_at)
        if window_words:
            self.window_words = int(window_words)

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        toks = []

        def one(p: Passage) -> float:
            body = {"model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"Question: {question}\n\nPassage ({p.path}):\n{p.text}"}]}
            r = post_json(self.url, body, headers)
            toks.append(int(r.get("usage", {}).get("prompt_tokens", 0)))
            try:
                top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except (KeyError, IndexError, TypeError):
                raise RerankerError("endpoint returned no logprobs; this backend needs logprobs support")
            yes = no = -1e9
            for t in top:
                w = t["token"].strip().lower()
                if w.startswith("yes"):
                    yes = max(yes, t["logprob"])
                elif w.startswith("no"):
                    no = max(no, t["logprob"])
            if yes <= -1e9 and no <= -1e9:
                return 0.0
            return math.exp(yes) / (math.exp(yes) + math.exp(no))

        out = pmap(one, passages, self.workers)
        self.last_tokens = sum(toks)
        return out
