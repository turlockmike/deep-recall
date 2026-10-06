"""Spend meter: an append-only JSON-lines ledger + per-query and rolling-24h caps.

Every paid reranker call batch is recorded as {ts, pid, backend, tokens, usd, caller}.
CALLER SPLIT (2026-10-01): one eval spent $1.02 of the shared rolling-24h cap and live recall ran unreranked
~24h (127/251 recalls first-stage). Rows now carry caller="live"|"eval"; each caller is capped against ITS OWN
rows only (eval -> eval_daily_cap_usd), so an eval can never starve live rerank. Legacy rows with no caller
count as live (conservative: never under-count the live cap). Before scoring, the
pipeline estimates the batch cost and refuses (BudgetExceeded -> fall back to first-stage order)
if it would cross either cap. A live cap of 0 disables the live cap; an EVAL cap of 0 REFUSES every eval call
(Mike 2026-10-01 20:03: "no more big jev tests, just use it for deeprecall calls only" -> eval_daily_cap_usd = 0).
SESSION SHARE (2026-10-06): one interactive PoE2 session fired 49 recalls in 50 min on 10-05 ($0.37 = 61% of
the $0.60 live cap); every later recall (incl. Mike-facing tg-delegate) ran unreranked 16:13-21:38. Rows now
carry session=CLAUDE_CODE_SESSION_ID; one session may fill at most session_share_frac of the live cap, so a
burst degrades only itself. Rows with no session never count toward any session share.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .rerankers.base import BudgetExceeded


def caller_of(row: dict) -> str:
    return "eval" if row.get("caller") == "eval" else "live"


class Budget:
    def __init__(self, ledger: Path, max_usd_per_query: float, daily_cap_usd: float, caller: str = "live",
                 session: str | None = None, session_share_frac: float = 0.0):
        self.ledger, self.per_query, self.daily = ledger, max_usd_per_query, daily_cap_usd
        self.caller = caller
        self.session, self.session_share = session or None, session_share_frac
        self.query_usd = 0.0
        self.ceiling = 1.0   # fraction of the daily cap this call may fill (widen passes run < 1.0; see recall.py)

    def day_usd(self, session: str | None = None) -> float:
        since, tot = time.time() - 86400, 0.0
        try:
            with open(self.ledger) as f:
                for ln in f:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if r.get("ts", 0) >= since and caller_of(r) == self.caller:
                        if session is None or r.get("session") == session:
                            tot += r.get("usd", 0.0)
        except FileNotFoundError:
            pass
        return tot

    def check(self, est_usd: float) -> None:
        if self.caller == "eval" and not self.daily:
            raise BudgetExceeded("eval jev spend is off (eval_daily_cap_usd = 0; Mike 2026-10-01 20:03, deeprecall live rerank only)")
        if est_usd <= 0:
            return
        if self.per_query and self.query_usd + est_usd > self.per_query:
            raise BudgetExceeded(f"query would cost ~${self.query_usd + est_usd:.4f} > per-query cap ${self.per_query:.2f}")
        if self.daily and self.ceiling < 1.0 and self.day_usd() + est_usd > self.daily * self.ceiling:
            raise BudgetExceeded(f"widen reserve: rolling-24h {self.caller} spend would pass "
                                 f"{self.ceiling:.0%} of ${self.daily:.2f}; rest kept for base reranks")
        if self.daily and self.day_usd() + est_usd > self.daily:
            raise BudgetExceeded(f"rolling-24h {self.caller} cap ${self.daily:.2f} reached (ledger {self.ledger})")
        if (self.caller == "live" and self.daily and self.session and 0 < self.session_share < 1
                and self.day_usd(self.session) + est_usd > self.daily * self.session_share):
            raise BudgetExceeded(f"session share: this session's rolling-24h {self.caller} spend would pass "
                                 f"{self.session_share:.0%} of ${self.daily:.2f}; rest kept for other callers")

    def record(self, backend: str, tokens: int, usd: float) -> None:
        self.query_usd += usd
        if usd <= 0 and tokens <= 0:
            return
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "pid": os.getpid(), "backend": backend,
                                    "tokens": tokens, "usd": round(usd, 8), "caller": self.caller,
                                    "session": self.session}) + "\n")
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
            "tokens_24h": sum(r.get("tokens", 0) for r in day), "usd_all_time": round(sum(r["usd"] for r in rows), 4),
            "usd_24h_by_caller": {c: round(sum(r["usd"] for r in day if caller_of(r) == c), 4) for c in ("live", "eval")}}
