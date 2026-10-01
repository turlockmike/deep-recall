"""Small evaluation harness: first-stage vs recall on a labeled question file.

Question file: JSON list or JSONL of {"q": "...", "gold": ["relative/path.md", ...]} where gold
lists every file that states the answer. Exact-path scoring undercounts when other files state
the same fact in other words; judge those by hand (or with a blind LLM judge) for headline numbers.
"""
from __future__ import annotations

import json
import sys

from .config import load
from .recall import Recaller


def _read(path: str) -> list[dict]:
    txt = open(path).read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def _experiment_ban() -> str | None:
    """Mike's jev-experiment ban (2026-09-29, ~/.config/jev/NO-EXPERIMENTS). An eval is an experiment that
    spends TypeSafe credits through deep-recall's OWN client, so the gate lives here, at the spend path,
    not only in the bash wrapper (auditor D427-V2-1). Fail closed: if jevlib can't be imported, refuse."""
    import os
    flag = os.path.expanduser("~/.config/jev/NO-EXPERIMENTS")
    if not os.path.exists(flag):
        return None
    try:
        sys.path.insert(0, os.path.expanduser("~/.local/lib/python"))
        import jevlib
        if jevlib.is_interactive():
            return None
    except Exception:
        pass
    return f"deeprecall eval REFUSED: jev experiments are banned outside a live session with Mike ({flag})"


def run(path: str, limit: int | None, top: int, max_usd: float, as_json: bool) -> int:
    ban = _experiment_ban()
    if ban:
        print(ban, file=sys.stderr)
        return 3
    qs = _read(path)[:limit] if limit else _read(path)
    cfg = load()
    rc = Recaller(cfg)
    rows, spent = [], 0.0
    for i, d in enumerate(qs, 1):
        gold = set(d["gold"])
        base = [p for p in rc.first_stage(d["q"], max(top, 10))]
        r = rc.recall(d["q"], top=top)
        spent += r.usd
        got = [h.path for h in r.hits]
        rows.append({"q": d["q"], "base1": bool(base[:1]) and base[0] in gold, "base3": bool(gold & set(base[:top])),
                     "rec1": bool(got[:1]) and got[0] in gold, "rec3": bool(gold & set(got[:top])),
                     "mode": r.mode, "usd": r.usd, "secs": r.secs})
        b1 = sum(x["base1"] for x in rows); r1 = sum(x["rec1"] for x in rows)
        print(f"[{i}/{len(qs)}] first-stage@1 {b1}  recall@1 {r1}  spent ${spent:.4f}", file=sys.stderr, flush=True)
        if spent >= max_usd:
            print(f"stopping: spend ${spent:.4f} reached --max-usd {max_usd}", file=sys.stderr)
            break
    n = len(rows)
    s = {k: sum(x[k] for x in rows) for k in ("base1", "base3", "rec1", "rec3")}
    out = {"n": n, "first_stage": {"hit@1": s["base1"], f"hit@{top}": s["base3"]},
           "recall": {"hit@1": s["rec1"], f"hit@{top}": s["rec3"]}, "usd": round(spent, 4),
           "mean_secs": round(sum(x["secs"] for x in rows) / max(n, 1), 2)}
    print(json.dumps(out if not as_json else {**out, "rows": rows}, indent=1))
    return 0
