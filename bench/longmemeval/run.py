#!/usr/bin/env python3
"""LongMemEval retrieval benchmark for deep-recall: does the loaded context hold every evidence session?

Each question gets its own index over its own haystack (one Markdown file per chat session). The script
runs first-stage search and reranked recall, and scores both against the dataset's answer_session_ids.
Abstention questions (question_id ending in _abs) are left out unless --include-abstention is given; the
dataset still labels sessions for them, and other published LongMemEval-S recall figures score all 500.

  python bench/longmemeval/run.py OUT_DIR --data longmemeval_s_cleaned.json [--n 470] [--backend jev]
                                  [--prompt-file bench/longmemeval/contrib-prompt.txt] [--question-kinds] [--max-usd 4]

Needs the deeprecall CLI on PATH (or $DEEPRECALL_BIN) and the backend's credentials (jev: $TYPESAFE_API_KEY).
Every deeprecall call runs with the other DEEPRECALL_* variables removed, so each question's own
./.deeprecall/config.toml is the only configuration in play.
Writes OUT_DIR/run.json (settings), OUT_DIR/sample.json, OUT_DIR/rows.ndjson (one line per question) and
OUT_DIR/summary.json, and prints the summary on stdout. Progress lines go to stderr. Re-running with the
same OUT_DIR and settings resumes; different settings are refused.
Exit 0 on success, 1 on a failed run (CLI error, empty index, reranker fell back to search order),
2 on bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BIN = os.environ.get("DEEPRECALL_BIN", "deeprecall")
PROMPT_BACKENDS = {"jev"}  # backends whose [reranker] table takes a `prompt`


def child_env() -> dict:
    """The environment for deeprecall calls: everything except DEEPRECALL_* overrides of config lookup."""
    return {k: v for k, v in os.environ.items() if not k.startswith("DEEPRECALL_")}


def check_prompt(prompt: str, backend: str) -> str | None:
    """None when the prompt is usable, else the reason it is not."""
    if backend not in PROMPT_BACKENDS:
        return f"--prompt-file applies to {sorted(PROMPT_BACKENDS)}, not {backend!r}"
    if "{q}" not in prompt:
        return "prompt needs a {q} placeholder for the question"
    try:
        prompt.format(q="x")
    except (KeyError, IndexError, ValueError) as e:
        return f"prompt has a format field other than {{q}} ({e!r}); write literal braces as {{{{ }}}}"
    return None


def check_recall(qid: str, recall: dict, search: list) -> None:
    """Raise when a recall result cannot be scored as a reranked result."""
    if recall["pool"] == 0 or not search:
        raise RuntimeError(f"{qid}: empty index or pool (pool={recall['pool']}, search hits={len(search)})")
    if not recall["mode"].startswith("rerank"):
        raise RuntimeError(f"{qid}: reranker did not run (mode={recall['mode']!r}, note={recall['note']!r})")


def answerable(questions: list[dict]) -> list[dict]:
    return [q for q in questions if not q["question_id"].endswith("_abs")]


def sample(questions: list[dict], n: int, seed: int = 0, include_abstention: bool = False) -> list[dict]:
    """A stratified sample of n questions, proportional to question_type, reproducible by seed."""
    pool = questions if include_abstention else answerable(questions)
    if not 0 < n <= len(pool):
        raise ValueError(f"n must be between 1 and {len(pool)} questions, got {n}")
    by = defaultdict(list)
    for q in pool:
        by[q["question_type"]].append(q)
    quota = {t: max(1, round(n * len(v) / len(pool))) for t, v in by.items()}
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(by, key=lambda t: len(by[t]) - quota[t])] += 1
    rng = random.Random(seed)
    out = []
    for t in sorted(by):
        out += rng.sample(sorted(by[t], key=lambda q: q["question_id"]), quota[t])
    return out


def score(ranked: list[str], gold: set[str]) -> dict:
    """Hit metrics for one ranked list of session files against the evidence set."""
    if not gold:
        raise ValueError("gold evidence set is empty")
    return {"any@1": bool(ranked[:1]) and ranked[0] in gold, "any@5": bool(gold & set(ranked[:5])),
            "all@5": gold <= set(ranked[:5]), "all@10": gold <= set(ranked[:10])}


def blind_name(qid: str, sid: str) -> str:
    """A file name that carries no label: LongMemEval names evidence sessions `answer_*`, and the name reaches
    the index (title), every reranker window (heading) and the reranker's state (`path`)."""
    return "s" + hashlib.sha256(f"{qid}:{sid}".encode()).hexdigest()[:10]


def session_markdown(sid: str, date: str, turns: list[dict]) -> str:
    return f"# Session {sid}\n\nDate: {date}\n\n" + "\n\n".join(f"**{t['role']}:** {t['content']}" for t in turns)


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, env=child_env(), capture_output=True, text=True)
    except OSError as e:
        raise RuntimeError(f"cannot run {cmd[0]}: {e}") from e
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} exit {p.returncode}: {p.stderr[-800:]}")
    return p.stdout


def _index(d: Path, qid: str) -> None:
    try:
        _run([BIN, "index", "--quiet"], d)
    except RuntimeError as e:
        # onnxruntime on macOS can abort (SIGABRT, exit -6) during interpreter shutdown after the index is
        # written; the rebuild is incremental, so one retry either confirms the index or surfaces a real error.
        if "exit -6" not in str(e):
            raise
        print(json.dumps({"event": "index_retry", "qid": qid, "err": str(e)[-160:]}), file=sys.stderr, flush=True)
        _run([BIN, "index", "--quiet"], d)


def configure(base: str, prompt: str, question_kinds: bool) -> str:
    """`deeprecall init`'s config with the benchmark's reranker prompt and question-kind setting applied."""
    if prompt:
        if "[reranker]\n" not in base or "prompt =" in base:
            raise ValueError("unexpected config layout, cannot set the reranker prompt")
        # a JSON string is a valid TOML basic string: quotes, backslashes and newlines arrive escaped
        base = base.replace("[reranker]\n", f"[reranker]\nprompt = {json.dumps(prompt)}\n", 1)
    if question_kinds:
        if "question_kinds = false" not in base:
            raise ValueError("unexpected config layout, cannot turn on question_kinds")
        base = base.replace("question_kinds = false", "question_kinds = true", 1)
    return base


def one(q: dict, out: Path, backend: str, prompt: str, question_kinds: bool = False) -> dict:
    d = out / "haystacks" / q["question_id"]
    notes = d / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    name = {sid: blind_name(q["question_id"], sid) for sid in q["haystack_session_ids"]}
    for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        (notes / f"{name[sid]}.md").write_text(session_markdown(name[sid], date, turns))
    cfg = d / ".deeprecall" / "config.toml"
    if not cfg.exists():
        _run([BIN, "init", "--root", str(notes), "--backend", backend, "--here"], d)
        try:
            cfg.write_text(configure(cfg.read_text(), prompt, question_kinds))
        except ValueError as e:
            cfg.unlink()
            raise RuntimeError(f"{cfg}: {e}") from e
    t0 = time.time()
    _index(d, q["question_id"])
    t_index = time.time() - t0
    question = q["question"].rstrip()
    if not question.endswith("?"):
        question += "?"  # deep-recall only reranks question-shaped queries
    s = json.loads(_run([BIN, "search", question, "-k", "10", "--json"], d))
    r = json.loads(_run([BIN, "recall", question, "--top", "10", "--backend", backend, "--no-passage", "--json"], d))
    check_recall(q["question_id"], r, s)
    gold = {f"{name[g]}.md" for g in q["answer_session_ids"]}
    sp, rp = [x["path"] for x in s], [x["path"] for x in r["results"]]
    return {"qid": q["question_id"], "type": q["question_type"], "gold": sorted(gold), "search": sp, "recall": rp,
            "scores": [x["score"] for x in r["results"]], "search_score": score(sp, gold), "recall_score": score(rp, gold),
            "mode": r["mode"], "note": r["note"], "kind": r.get("kind"), "pool": r["pool"], "widened": r["widened"], "usd": r["usd"],
            "tokens": r["tokens"], "secs_recall": r["secs"], "secs_index": round(t_index, 1)}


def summarize(rows: list[dict], backend: str) -> dict:
    n = len(rows)
    tot = lambda side, k: sum(r[f"{side}_score"][k] for r in rows)
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        b = by[r["type"]]
        b["n"] += 1
        for k in ("any@1", "all@5"):
            b[f"search {k}"] += r["search_score"][k]
            b[f"recall {k}"] += r["recall_score"][k]
    usd = sum(r["usd"] for r in rows)
    return {"n": n, "backend": backend,
            "search": {k: tot("search", k) for k in ("any@1", "any@5", "all@5", "all@10")},
            "recall": {k: tot("recall", k) for k in ("any@1", "any@5", "all@5", "all@10")},
            "widened": sum(r["widened"] for r in rows), "fail_open": sum(not r["mode"].startswith("rerank") for r in rows),
            "usd": round(usd, 4), "usd_per_q": round(usd / max(n, 1), 5), "tokens": sum(r.get("tokens", 0) for r in rows),
            "mean_secs_recall": round(sum(r["secs_recall"] for r in rows) / max(n, 1), 1),
            "by_type": {t: dict(v) for t, v in sorted(by.items())}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="run directory (created; re-use it to resume)")
    ap.add_argument("--data", required=True, help="LongMemEval JSON, e.g. longmemeval_s_cleaned.json")
    ap.add_argument("--n", type=int, default=50, help="answerable questions to sample (470 = all of LongMemEval-S)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-abstention", action="store_true",
                    help="also score the 30 abstention questions (500 = all of LongMemEval-S)")
    ap.add_argument("--backend", default="jev")
    ap.add_argument("--prompt-file", help="jev prompt override; must contain {q}")
    ap.add_argument("--question-kinds", action="store_true",
                    help="classify each question first: its kind picks the prompt, and aggregate/temporal always widen (jev)")
    ap.add_argument("--max-usd", type=float, default=1.0, help="stop before the next question once spend reaches this")
    a = ap.parse_args(argv)
    prompt = ""
    if a.prompt_file:
        prompt = Path(a.prompt_file).read_text().strip()
        bad = check_prompt(prompt, a.backend)
        if bad:
            print(json.dumps({"error": bad}), file=sys.stderr)
            return 2
    if a.question_kinds and a.backend not in PROMPT_BACKENDS:
        print(json.dumps({"error": f"--question-kinds applies to {sorted(PROMPT_BACKENDS)}, not {a.backend!r}"}), file=sys.stderr)
        return 2
    try:
        qs = sample(json.load(open(a.data)), a.n, a.seed, a.include_abstention)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"error": f"cannot load {a.data}: {e}"}), file=sys.stderr)
        return 2
    out = Path(a.out).resolve()
    settings = {"data": Path(a.data).name, "n": a.n, "seed": a.seed, "backend": a.backend, "prompt": prompt,
                "include_abstention": a.include_abstention, "blind_ids": True, "question_kinds": a.question_kinds}
    run_f = out / "run.json"
    if run_f.exists() and json.loads(run_f.read_text()) != settings:
        print(json.dumps({"error": f"{out} was started with other settings; use a new directory",
                          "existing": json.loads(run_f.read_text()), "requested": settings}), file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    run_f.write_text(json.dumps(settings, indent=1) + "\n")
    (out / "sample.json").write_text(json.dumps([q["question_id"] for q in qs], indent=1))
    rows_f = out / "rows.ndjson"
    wanted = {q["question_id"] for q in qs}
    rows = [json.loads(l) for l in rows_f.read_text().splitlines()] if rows_f.exists() else []
    rows = [r for r in rows if r["qid"] in wanted]
    done, spent = {r["qid"] for r in rows}, sum(r["usd"] for r in rows)
    for i, q in enumerate(qs, 1):
        if q["question_id"] in done:
            continue
        if spent >= a.max_usd:
            print(json.dumps({"event": "stop", "reason": f"spend ${spent:.4f} reached --max-usd {a.max_usd}"}), file=sys.stderr)
            break
        try:
            row = one(q, out, a.backend, prompt, a.question_kinds)
        except RuntimeError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            return 1
        spent += row["usd"]
        rows.append(row)
        with rows_f.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"i": i, "qid": row["qid"], "type": row["type"], "all@5": row["recall_score"]["all@5"],
                          "mode": row["mode"], "kind": row["kind"], "usd": row["usd"], "spent": round(spent, 4)}), file=sys.stderr, flush=True)
    summary = json.dumps(summarize(rows, a.backend), indent=1)
    (out / "summary.json").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
