"""Run any executable as the reranker. Language-agnostic escape hatch.

The command receives on stdin:
    {"question": "...", "passages": [{"path": "...", "heading": "...", "text": "..."}, ...]}
and must print on stdout a JSON list of floats in [0, 1] (one per passage, same order), or
{"scores": [...], "tokens": <int, optional>}.

Config:
    [reranker]
    backend = "command"
    command = ["python3", "/path/to/my_scorer.py"]
    window_words = 600
    widen_at = 0.7
    usd_per_mtok = 0.0
    batch = 64          # passages per invocation
"""
from __future__ import annotations

import json
import shlex
import subprocess

from .base import Passage, Reranker, RerankerError


class CommandReranker(Reranker):
    name = "command"

    def __init__(self, command, window_words: int = 600, widen_at: float = 0.7, usd_per_mtok: float = 0.0,
                 batch: int = 64, timeout: float = 300, **_):
        self.cmd = shlex.split(command) if isinstance(command, str) else list(command)
        self.window_words, self.widen_at = int(window_words), float(widen_at)
        self.usd_per_token, self.batch, self.timeout = usd_per_mtok / 1e6, int(batch), timeout

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        out, toks = [], 0
        for i in range(0, len(passages), self.batch):
            chunk = passages[i:i + self.batch]
            payload = json.dumps({"question": question,
                                  "passages": [p.__dict__ for p in chunk]})
            r = subprocess.run(self.cmd, input=payload, capture_output=True, text=True, timeout=self.timeout)
            if r.returncode != 0:
                raise RerankerError(f"command failed rc={r.returncode}: {r.stderr[:300]}")
            try:
                res = json.loads(r.stdout)
            except json.JSONDecodeError:
                raise RerankerError(f"command printed non-JSON: {r.stdout[:200]!r}")
            scores = res.get("scores") if isinstance(res, dict) else res
            if not isinstance(scores, list) or len(scores) != len(chunk):
                raise RerankerError(f"command must return {len(chunk)} scores")
            toks += int(res.get("tokens", 0)) if isinstance(res, dict) else 0
            out += [float(s) for s in scores]
        self.last_tokens = toks
        return out
