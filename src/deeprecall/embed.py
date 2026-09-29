"""Embedders. Default: fastembed (ONNX, CPU, no PyTorch). `hash:<dim>` is a deterministic
bag-of-words hashing embedder for tests and offline smoke runs (no model download)."""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache


class HashEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FastEmbedder:
    def __init__(self, model: str, batch_size: int = 64):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model)
        self.batch_size = batch_size
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        # sort by length so each batch pads to similar sizes (~2x less compute), then restore order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = list(self.model.embed([texts[i] for i in order], batch_size=self.batch_size))
        out: list[list[float]] = [[] for _ in texts]
        for i, v in zip(order, vecs):
            out[i] = list(map(float, v))
        return out


@lru_cache(maxsize=4)
def get_embedder(model: str):
    if model.startswith("hash"):
        return HashEmbedder(int(model.split(":")[1]) if ":" in model else 64)
    return FastEmbedder(model)


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def mean(vs: list[list[float]]) -> list[float]:
    d = len(vs[0])
    return normalize([sum(v[i] for v in vs) / len(vs) for i in range(d)])
