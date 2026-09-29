"""Local cross-encoder reranker (fastembed / ONNX, CPU). Free, private, no API key.

Scores are relevance logits; sigmoid maps them to (0, 1). Cross-encoders judge *relevance*,
not strictly *answerability*, so they are weaker than an answerability-prompted model on
buried-fact questions, but they are a solid zero-cost default.
Models: Xenova/ms-marco-MiniLM-L-6-v2 (fast, default), Xenova/ms-marco-MiniLM-L-12-v2,
BAAI/bge-reranker-base, jinaai/jina-reranker-v1-turbo-en, jinaai/jina-reranker-v2-base-multilingual.
"""
from __future__ import annotations

import math

from .base import Passage, Reranker


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"
    window_words = 300          # these models read <= 512 tokens
    widen_at = 0.5
    prefilter_per_file = 3      # CPU-bound: score each file's 3 most question-like windows only

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", batch_size: int = 32,
                 widen_at: float | None = None, window_words: int | None = None,
                 prefilter_per_file: int | None = None, threads: int | None = None, **_):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model, threads=threads)
        self.batch_size = batch_size
        if widen_at is not None:
            self.widen_at = float(widen_at)
        if window_words:
            self.window_words = int(window_words)
        if prefilter_per_file is not None:
            self.prefilter_per_file = int(prefilter_per_file)

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        docs = [p.text for p in passages]
        logits = list(self.model.rerank(question, docs, batch_size=self.batch_size))
        self.last_tokens = 0
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
