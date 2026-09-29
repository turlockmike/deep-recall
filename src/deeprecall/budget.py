"""Spend meter: an append-only JSON-lines ledger + per-query and rolling-24h caps.

Every paid reranker call batch is recorded as {ts, backend, tokens, usd}. Before scoring, the
pipeline estimates the batch cost and refuses (BudgetExceeded -> fall back to first-stage order)
if it would cross either cap. Caps of 0 disable.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .rerankers.base import BudgetExceeded


class Budget:
    def __init__(self, ledger: Path, max_usd_per_query: float, daily_cap_usd: float):
        self.ledger, self.per_query, self.daily = ledger, max_usd_per_query, daily_cap_usd
        self.query_usd = 0.0

    def day_usd(self) -> float:
        since, tot = time.time() - 86400, 0.0
        try:
            with open(self.ledger) as f:
                for ln in f:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if r.get("ts", 0) >= since:
                        tot += r.get("usd", 0.0)
        except FileNotFoundError:
            pass
        return tot

    def check(self, est_usd: float) -> None:
        if est_usd <= 0:
            return
        if self.per_query and self.query_usd + est_usd > self.per_query:
            raise BudgetExceeded(f"query would cost ~${self.query_usd + est_usd:.4f} > per-query cap ${self.per_query:.2f}")
        if self.daily and self.day_usd() + est_usd > self.daily:
            raise BudgetExceeded(f"rolling-24h cap ${self.daily:.2f} reached (ledger {self.ledger})")

    def record(self, backend: str, tokens: int, usd: float) -> None:
        self.query_usd += usd
        if usd <= 0 and tokens <= 0:
            return
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "pid": os.getpid(), "backend": backend,
                                    "tokens": tokens, "usd": round(usd, 8)}) + "\n")
        except OSError:
            pass


def summary(ledger: Path) -> dict:
    since, rows = time.time() - 86400, []
    try:
        rows = [json.loads(l) for l in open(ledger) if l.strip()]
    except FileNotFoundError:
        pass
    day = [r for r in rows if r.get("ts", 0) >= since]
    return {"ledger": str(ledger), "calls_24h": len(day), "usd_24h": round(sum(r["usd"] for r in day), 4),
            "tokens_24h": sum(r.get("tokens", 0) for r in day), "usd_all_time": round(sum(r["usd"] for r in rows), 4)}
