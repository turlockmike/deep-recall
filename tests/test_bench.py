"""Offline tests for the LongMemEval harness's pure logic: sampling, scoring, summaries, argument errors."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "lme_run", Path(__file__).resolve().parent.parent / "bench" / "longmemeval" / "run.py")
lme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lme)


def _qs(counts: dict[str, int], abstain: int = 0) -> list[dict]:
    out = [{"question_id": f"{t}-{i}", "question_type": t} for t, c in counts.items() for i in range(c)]
    return out + [{"question_id": f"x{i}_abs", "question_type": "multi-session"} for i in range(abstain)]


def test_sample_is_stratified_exact_size_and_skips_abstention():
    qs = _qs({"a": 60, "b": 30, "c": 10}, abstain=5)
    s = lme.sample(qs, 10)
    assert len(s) == 10 and len({q["question_id"] for q in s}) == 10
    assert not any(q["question_id"].endswith("_abs") for q in s)
    assert Counter(q["question_type"] for q in s) == {"a": 6, "b": 3, "c": 1}


def test_sample_keeps_every_type_and_is_reproducible():
    qs = _qs({"a": 97, "b": 2, "c": 1})
    s = lme.sample(qs, 5)
    assert set(q["question_type"] for q in s) == {"a", "b", "c"} and len(s) == 5
    assert [q["question_id"] for q in s] == [q["question_id"] for q in lme.sample(qs, 5)]
    assert [q["question_id"] for q in s] != [q["question_id"] for q in lme.sample(qs, 5, seed=1)]


def test_sample_can_include_abstention():
    qs = _qs({"a": 7, "b": 3}, abstain=2)
    s = lme.sample(qs, 12, include_abstention=True)
    assert len(s) == 12 and sum(q["question_id"].endswith("_abs") for q in s) == 2


def test_sample_all_returns_every_answerable_question():
    qs = _qs({"a": 7, "b": 3}, abstain=2)
    assert len(lme.sample(qs, 10)) == 10


def test_sample_tops_up_when_rounding_undershoots():
    s = lme.sample(_qs({"a": 5, "b": 5, "c": 5}), 4)
    assert len(s) == 4 and set(q["question_type"] for q in s) == {"a", "b", "c"}


@pytest.mark.parametrize("n", [0, 11])
def test_sample_rejects_out_of_range_n(n):
    with pytest.raises(ValueError, match="between 1 and 10 questions"):
        lme.sample(_qs({"a": 7, "b": 3}), n)


def test_score_distinguishes_any_from_all():
    gold = {"g1.md", "g2.md"}
    s = lme.score(["g1.md", "x.md", "y.md", "z.md", "w.md", "g2.md"], gold)
    assert s == {"any@1": True, "any@5": True, "all@5": False, "all@10": True}
    s = lme.score(["x.md", "g1.md", "g2.md"], gold)
    assert s == {"any@1": False, "any@5": True, "all@5": True, "all@10": True}
    assert lme.score([], gold) == {"any@1": False, "any@5": False, "all@5": False, "all@10": False}


def test_score_rejects_empty_gold():
    with pytest.raises(ValueError):
        lme.score(["a.md"], set())


def _row(t, rec, sea, usd=0.01, mode="rerank:jev", widened=False):
    return {"type": t, "recall_score": rec, "search_score": sea, "usd": usd, "tokens": 100, "mode": mode,
            "widened": widened, "secs_recall": 2.0}


def test_summarize_counts_per_side_type_failopen_and_cost():
    hit = {"any@1": True, "any@5": True, "all@5": True, "all@10": True}
    miss = {"any@1": False, "any@5": True, "all@5": False, "all@10": True}
    rows = [_row("a", hit, miss, widened=True), _row("a", miss, miss), _row("b", hit, hit, mode="first-stage")]
    s = lme.summarize(rows, "jev")
    assert s["n"] == 3 and s["recall"]["all@5"] == 2 and s["search"]["all@5"] == 1 and s["recall"]["all@10"] == 3
    assert s["by_type"]["a"] == {"n": 2, "search any@1": 0, "recall any@1": 1, "search all@5": 0, "recall all@5": 1}
    assert s["fail_open"] == 1 and s["widened"] == 1 and s["usd"] == 0.03 and s["usd_per_q"] == 0.01


def test_session_markdown_carries_id_date_and_roles():
    md = lme.session_markdown("s1", "2023/05/23 (Tue) 01:43", [{"role": "user", "content": "I drove 5 hours."}])
    assert md.startswith("# Session s1\n\nDate: 2023/05/23 (Tue) 01:43\n\n") and "**user:** I drove 5 hours." in md


@pytest.mark.parametrize("prompt,backend,why", [
    ("no placeholder here", "jev", "placeholder"),
    ("about {q} and {other}", "jev", "format field"),
    ("unbalanced {q} {", "jev", "format field"),
    ("fine {q}", "cross-encoder", "applies to"),
])
def test_check_prompt_rejects(prompt, backend, why):
    assert why in lme.check_prompt(prompt, backend)


def test_check_prompt_accepts_quotes_braces_and_newlines():
    assert lme.check_prompt('Does it answer "{q}"?\nIt\'s fine to write {{literal}} braces.', "jev") is None


def _data(tmp_path, counts={"a": 3}):
    qs = [{**q, "question": "Which plan?", "answer_session_ids": ["s1"], "haystack_session_ids": ["s1"],
           "haystack_dates": ["2023/01/01 (Sun) 10:00"], "haystack_sessions": [[{"role": "user", "content": "hi"}]]}
          for q in _qs(counts)]
    data = tmp_path / "d.json"
    data.write_text(json.dumps(qs))
    return data


def test_main_rejects_bad_prompt_before_touching_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(lme, "BIN", str(tmp_path / "no-such-deeprecall"))
    pf = tmp_path / "p.txt"
    pf.write_text("no placeholder here")
    assert lme.main([str(tmp_path / "out"), "--data", str(_data(tmp_path)), "--n", "1", "--prompt-file", str(pf)]) == 2
    assert not (tmp_path / "out").exists()


def test_main_exits_1_when_the_cli_cannot_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lme, "BIN", str(tmp_path / "no-such-deeprecall"))
    assert lme.main([str(tmp_path / "out"), "--data", str(_data(tmp_path)), "--n", "1"]) == 1


def test_main_refuses_to_resume_with_other_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(lme, "BIN", str(tmp_path / "no-such-deeprecall"))
    out, data = tmp_path / "out", _data(tmp_path)
    assert lme.main([str(out), "--data", str(data), "--n", "1"]) == 1          # records run.json, then fails
    assert json.loads((out / "run.json").read_text())["seed"] == 0
    assert lme.main([str(out), "--data", str(data), "--n", "1", "--seed", "1"]) == 2
    assert lme.main([str(out), "--data", str(data), "--n", "2"]) == 2
    assert lme.main([str(out), "--data", str(data), "--n", "1"]) == 1          # same settings resume


def test_main_summarizes_only_the_current_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(lme, "BIN", str(tmp_path / "no-such-deeprecall"))
    out, data = tmp_path / "out", _data(tmp_path)
    hit = {"any@1": True, "any@5": True, "all@5": True, "all@10": True}
    out.mkdir()
    (out / "run.json").write_text(json.dumps({"data": "d.json", "n": 3, "seed": 0, "backend": "jev", "prompt": "",
                                             "include_abstention": False, "blind_ids": True}))
    rows = [{**_row("a", hit, hit), "qid": q} for q in ("a-0", "a-1", "a-2", "stale-9")]
    (out / "rows.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert lme.main([str(out), "--data", str(data), "--n", "3"]) == 0
    assert json.loads((out / "summary.json").read_text())["n"] == 3


def test_child_env_drops_config_overrides(monkeypatch):
    for k in ("DEEPRECALL_CONFIG", "DEEPRECALL_INDEX", "DEEPRECALL_ROOTS", "DEEPRECALL_RERANKER"):
        monkeypatch.setenv(k, "/elsewhere")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    env = lme.child_env()
    assert not any(k.startswith("DEEPRECALL_") for k in env) and env["TYPESAFE_API_KEY"] == "k"


@pytest.mark.parametrize("recall,search,why", [
    ({"pool": 0, "mode": "rerank:jev", "note": ""}, ["a.md"], "empty"),
    ({"pool": 5, "mode": "rerank:jev", "note": ""}, [], "empty"),
    ({"pool": 5, "mode": "first-stage (reranker unavailable)", "note": "jev: set $TYPESAFE_API_KEY"}, ["a.md"], "did not run"),
])
def test_check_recall_rejects_unscorable_results(recall, search, why):
    with pytest.raises(RuntimeError, match=why):
        lme.check_recall("q1", recall, search)


def test_check_recall_accepts_a_reranked_result():
    lme.check_recall("q1", {"pool": 20, "mode": "rerank:jev", "note": ""}, ["a.md"])


def test_main_rejects_missing_data_and_bad_n(tmp_path):
    assert lme.main([str(tmp_path / "out"), "--data", str(tmp_path / "nope.json")]) == 2
    assert lme.main([str(tmp_path / "out"), "--data", str(_data(tmp_path)), "--n", "4"]) == 2


_cspec = importlib.util.spec_from_file_location(
    "lme_charts", Path(__file__).resolve().parent.parent / "bench" / "longmemeval" / "charts.py")
charts = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(charts)


def test_charts_render_the_committed_results_and_match_the_checked_in_svgs():
    res = Path(charts.HERE) / "results"
    lme = json.loads((res / "s-470-contrib-prompt" / "summary.json").read_text())
    lme500 = json.loads((res / "s-500-contrib-prompt" / "summary.json").read_text())
    lift, miss = charts.reranker_lift(lme, lme500), charts.misses(lme500)
    assert f"{lme['recall']['all@5'] / lme['n'] * 100:.1f}%" in lift
    assert f">{lme500['n'] - lme500['recall']['any@5']} of {lme500['n']}<" in miss
    assert (charts.OUT / "reranker-lift.svg").read_text() == lift
    assert (charts.OUT / "longmemeval-misses.svg").read_text() == miss


def test_blind_name_hides_the_evidence_label_and_is_stable():
    n = lme.blind_name("q1", "answer_526354c8_1")
    assert "answer" not in n and n == lme.blind_name("q1", "answer_526354c8_1")
    assert n != lme.blind_name("q2", "answer_526354c8_1")
