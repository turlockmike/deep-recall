"""Reranker plugin interface.

A reranker reads (question, passage) pairs and returns, for each passage, a score in [0, 1]:
the probability that the passage *states the answer* to the question. That one contract is
all deeprecall needs; any model that can produce it plugs in.

Write your own:

    from deeprecall.rerankers.base import Reranker, Passage

    class MyReranker(Reranker):
        name = "mine"
        window_words = 400        # passage size you want (words)
        widen_at = 0.6            # best score below this => deeprecall widens the candidate pool once
        usd_per_token = 0.0       # for the spend meter (input tokens, estimated as chars/3 before a call)

        def __init__(self, **options):   # options = the [reranker] table from config.toml
            ...

        def score(self, question: str, passages: list[Passage]) -> list[float]:
            ...

Register it in your package's pyproject.toml:

    [project.entry-points."deeprecall.rerankers"]
    mine = "mypkg.module:MyReranker"

or point the config straight at it:  [reranker] backend = "mypkg.module:MyReranker"
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class Passage:
    path: str
    heading: str
    text: str


class Reranker:
    name = "base"
    window_words = 600
    widen_at = 0.5
    usd_per_token = 0.0
    prefilter_per_file = 0     # >0: keep only this many windows per file (by word overlap) before scoring
    last_tokens = 0            # set by score(): billed/used input tokens of the last call batch

    def __init__(self, **options):
        self.options = options

    def estimate_tokens(self, question: str, passages: list[Passage]) -> int:
        return sum(len(p.text) + len(question) + 200 for p in passages) // 3

    def score(self, question: str, passages: list[Passage]) -> list[float]:  # pragma: no cover
        raise NotImplementedError


class NoReranker(Reranker):
    """Keeps first-stage order (scores descend with rank)."""
    name = "none"
    widen_at = 0.0

    def score(self, question, passages):
        n = len(passages)
        return [1.0 - i / max(n, 1) for i in range(n)]


class RerankerError(RuntimeError):
    pass


class BudgetExceeded(RerankerError):
    pass


BUILTIN = {
    "none": "deeprecall.rerankers.base:NoReranker",
    "cross-encoder": "deeprecall.rerankers.crossencoder:CrossEncoderReranker",
    "jev": "deeprecall.rerankers.jev:JevReranker",
    "openai": "deeprecall.rerankers.openai_compat:OpenAIYesNoReranker",
    "command": "deeprecall.rerankers.command:CommandReranker",
}


def _import(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def available() -> dict[str, str]:
    out = dict(BUILTIN)
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="deeprecall.rerankers"):
            out.setdefault(ep.name, ep.value)
    except Exception:
        pass
    return out


def load(options: dict) -> Reranker:
    opts = dict(options)
    backend = opts.pop("backend", "cross-encoder")
    spec = available().get(backend) or (backend if ":" in backend else None)
    if not spec:
        raise RerankerError(f"unknown reranker backend {backend!r}; available: {', '.join(sorted(available()))}")
    return _import(spec)(**opts)
