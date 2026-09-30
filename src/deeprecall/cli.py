"""deeprecall command line.

  deeprecall init [--root DIR ...] [--here]     write a config (global, or ./.deeprecall/ with --here)
  deeprecall index [--full] [--quiet]           build / incrementally update the index
  deeprecall search "terms" [-k 10]             first-stage hybrid search (free, fast)
  deeprecall recall "question?" [--top 5]       answer-aware recall (reranked), prints the winning section
  deeprecall status                             config, index size, reranker, spend
  deeprecall rerankers                          list available reranker backends
  deeprecall report [--days 7]                  usage: queries, confidence, widen/fallback rate, latency, spend
  deeprecall eval questions.jsonl [--limit N]   hit@1 / hit@3 for first-stage vs recall
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import index as idx
from .budget import summary
from .config import TEMPLATE, find_config_file, load


def _print_recall(r, as_json: bool, show_passage: bool, cap: int = 3000) -> None:
    if as_json:
        print(json.dumps({"query": r.query, "mode": r.mode, "pool": r.pool, "widened": r.widened, "usd": r.usd,
                          "tokens": r.tokens, "secs": r.secs, "note": r.note, "kind": r.kind,
                          "results": [{"path": h.path, "score": h.score, "section": h.section} for h in r.hits],
                          "passage": r.hits[0].passage if r.hits else ""}, indent=1))
        return
    for i, h in enumerate(r.hits, 1):
        s = f"{h.score:.2f}" if h.score is not None else "  - "
        print(f"{i}. {s}  {h.path}" + (f"\n         § {h.section[:100]}" if h.section else ""))
    top = r.hits[0] if r.hits else None
    wa = r.extra.get("widen_at", 0.5)
    if show_passage and top and top.passage and top.score is not None and top.score >= wa:
        body = top.passage[:cap] + (" …[truncated; open the file]" if len(top.passage) > cap else "")
        print(f"\n--- #1 winning section ({top.path}) ---\n{body}")
    meta = f"[{r.mode} · pool {r.pool} · ${r.usd:.4f} · {r.secs:.1f}s"
    meta += f" · {r.kind}" if r.kind else ""
    meta += " · widened" if r.widened else ""
    meta += f" · {r.note}" if r.note else ""
    print(meta + "]", file=sys.stderr)
    if top and top.score is not None and top.score < wa:
        print("[low confidence: no file clearly states the answer]", file=sys.stderr)


def cmd_init(a) -> int:
    target = (Path.cwd() / ".deeprecall" / "config.toml") if a.here else (Path.home() / ".config" / "deeprecall" / "config.toml")
    if target.exists() and not a.force:
        print(f"{target} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    text = TEMPLATE
    if a.root:
        roots = ", ".join(json.dumps(str(Path(r).expanduser().resolve())) for r in a.root)
        text = text.replace('roots = ["~/notes"]', f"roots = [{roots}]")
    if a.backend:
        text = text.replace('backend = "cross-encoder"', f'backend = "{a.backend}"')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"wrote {target}\nnext: deeprecall index")
    return 0


def cmd_index(a) -> int:
    cfg = load()
    if not cfg.roots:
        print("no roots configured: run `deeprecall init --root <notes dir>`", file=sys.stderr)
        return 2
    r = idx.build(cfg, full=a.full, quiet=a.quiet)
    if r.get("skipped"):
        print(f"index: {r['skipped']}", file=sys.stderr)
        return 0
    if not a.quiet or r["indexed"] or r["removed"]:
        print(f"indexed {r['indexed']} changed file(s), removed {r['removed']}, total {r['total']} ({r['secs']}s) -> {cfg.index}")
    return 0


def cmd_search(a) -> int:
    from .search import search
    cfg = load()
    res = search(cfg, a.query, a.k)
    if a.json:
        print(json.dumps([{"path": p, "score": round(s, 5)} for p, s in res], indent=1))
    else:
        for i, (p, s) in enumerate(res, 1):
            print(f"{i}. {s:.4f}  {p}")
    return 0


def cmd_recall(a) -> int:
    from .recall import Recaller
    cfg = load()
    if a.backend:
        cfg.reranker = {**cfg.reranker, "backend": a.backend}
    r = Recaller(cfg).recall(a.query, top=a.top, k=a.k, rerank=not a.no_rerank)
    _print_recall(r, a.json, not a.no_passage)
    return 0


def cmd_status(a) -> int:
    from .rerankers.base import available
    cfg = load()
    out = {"version": __version__, "config": str(cfg.source or find_config_file() or "(none: defaults)"),
           "roots": [str(r) for r in cfg.roots], "reranker": cfg.reranker, "index": idx.stats(cfg),
           "spend": summary(cfg.ledger), "caps": {"per_query_usd": cfg.max_usd_per_query, "daily_usd": cfg.daily_cap_usd},
           "rerankers_available": sorted(available())}
    print(json.dumps(out, indent=1))
    return 0


def cmd_rerankers(a) -> int:
    from .rerankers.base import available
    for k, v in sorted(available().items()):
        print(f"{k:15} {v}")
    return 0


def cmd_report(a) -> int:
    import statistics
    import time as _t
    cfg = load()
    since = _t.time() - a.days * 86400
    rows = []
    try:
        for ln in open(cfg.recall_log):
            r = json.loads(ln)
            try:
                ts = _t.mktime(_t.strptime(r["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
            except (KeyError, ValueError):
                continue
            if ts >= since:
                rows.append(r)
    except FileNotFoundError:
        pass
    rr = [r for r in rows if r["mode"].startswith("rerank")]
    wa = lambda r: r.get("widen_at") or 0.5
    top = lambda r: (r["top"][0][1] if r.get("top") and r["top"][0][1] is not None else None)
    secs = sorted(r["secs"] for r in rr) or [0]
    agree = [r for r in rr if r.get("first_stage_top3") and r.get("top")]
    out = {"days": a.days, "queries": len(rows),
           "modes": {m: sum(r["mode"] == m for r in rows) for m in sorted({r["mode"] for r in rows})},
           "reranked": len(rr),
           "confident_top1": sum(1 for r in rr if top(r) is not None and top(r) >= wa(r)),
           "low_confidence": sum(1 for r in rr if top(r) is not None and top(r) < wa(r)),
           "widened": sum(r.get("widened", False) for r in rr),
           "fallbacks": sum(r["mode"].startswith("first-stage") for r in rows),
           "top1_differs_from_search": sum(r["top"][0][0] != r["first_stage_top3"][0] for r in agree),
           "secs_p50": round(statistics.median(secs), 2), "secs_p90": round(secs[int(0.9 * (len(secs) - 1))], 2),
           "usd": round(sum(r.get("usd", 0) for r in rows), 4),
           "usd_per_reranked": round(sum(r.get("usd", 0) for r in rr) / max(len(rr), 1), 5)}
    print(json.dumps(out, indent=1))
    return 0


def cmd_eval(a) -> int:
    from .eval import run
    return run(a.file, a.limit, a.top, a.max_usd, a.json)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deeprecall", description="Answer-aware recall over Markdown notes.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a config")
    p.add_argument("--root", action="append", help="notes folder (repeatable)")
    p.add_argument("--backend", help="reranker backend (cross-encoder, jev, openai, command, none)")
    p.add_argument("--here", action="store_true", help="write ./.deeprecall/config.toml instead of the global one")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("index", help="build or update the index")
    p.add_argument("--full", action="store_true", help="rebuild from scratch")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("search", help="first-stage hybrid search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("recall", help="answer-aware recall")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("-k", type=int, default=None, help="first-stage depth (default from config)")
    p.add_argument("--backend", help="override the reranker backend for this call")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-passage", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_recall)
    p = sub.add_parser("status", help="show config, index, spend")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("rerankers", help="list reranker backends")
    p.set_defaults(fn=cmd_rerankers)
    p = sub.add_parser("report", help="usage summary from the recall log")
    p.add_argument("--days", type=float, default=7)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("eval", help="hit@1/hit@3 on a question file")
    p.add_argument("file", help='JSON list or JSONL of {"q": "...", "gold": ["path", ...]}')
    p.add_argument("--limit", type=int)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--max-usd", type=float, default=1.0, help="stop when reranker spend reaches this")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
