"""Offline tests: hash embedder + fake rerankers, no model downloads or network."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from deeprecall import index as idx
from deeprecall.budget import Budget
from deeprecall.config import load
from deeprecall.markdown import split_frontmatter, split_units, windows
from deeprecall.rare import RareIndex, terms
from deeprecall.recall import Recaller, is_question
from deeprecall.rerankers.base import BudgetExceeded, Passage, Reranker, load as load_reranker
from deeprecall.search import fts_query, rrf, search

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "notes"


@pytest.fixture()
def cfg(tmp_path):
    notes = tmp_path / "notes"
    shutil.copytree(EXAMPLES, notes)
    c = tmp_path / "config.toml"
    for junk in notes.rglob("*"):
        if junk.name.lower() == "index.md":
            junk.unlink()
    c.write_text(f'roots = ["{notes}"]\nindex = "{tmp_path / "state" / "index.db"}"\n'
                 '[embedding]\nmodel = "hash:64"\nchunk_words = 40\n[reranker]\nbackend = "none"\n')
    conf = load(c)
    idx.build(conf, quiet=True)
    return conf


class KeywordFake(Reranker):
    """Scores a passage by whether it contains a marker string."""
    name = "fake"
    window_words = 60
    widen_at = 0.7
    usd_per_token = 1e-6

    def __init__(self, marker="Saver Plus 12", **_):
        self.marker = marker

    def score(self, q, passages):
        self.last_tokens = sum(len(p.text) // 3 for p in passages)
        return [0.95 if self.marker in p.text else 0.1 for p in passages]


def test_frontmatter_and_units():
    fm, body = split_frontmatter("---\ntitle: T\nsummary: s\n---\n# A\nx\n## B\ny y\n")
    assert fm["title"] == "T" and body.startswith("# A")
    u = split_units(body, 350)
    assert [h for h, _, _ in u] == ["A", "A > B"]


def test_fenced_headings_ignored():
    u = split_units("# A\n```\n# not a heading\n```\ntext\n")
    assert len(u) == 1 and u[0][0] == "A"


def test_windows_bounded():
    body = "\n".join(f"## H{i}\n" + "w " * 300 for i in range(100))
    ws = windows(body, 600, 24)
    assert len(ws) <= 25 and sum(len(t.split()) for _, t in ws) >= 30000


def test_terms():
    t = terms('Why did the ingest-sync-daemon fail and what is KXGDP or "Pearl Ring" in BLS data?')
    assert {"ingest-sync-daemon", "KXGDP", "BLS", "Pearl Ring"} <= set(t)
    assert "Why" not in t and "data" not in t


def test_question_gate():
    assert is_question("which plan did we pick")
    assert is_question("electricity plan?")
    assert not is_question("electricity plan")


def test_fts_query_is_safe():
    assert fts_query('what, exactly: "kk" (fusion) AND NOT xy-zz?') == '"what" OR "exactly" OR "kk" OR "fusion" OR "xy" OR "zz"'


def test_rrf():
    fused = dict(rrf([(["a", "b"], 1.0), (["b", "c"], 0.5)], 60))
    assert fused["b"] > fused["a"] > fused["c"]


def test_index_incremental(cfg):
    s = idx.stats(cfg)
    assert s["docs"] == 9 and s["sections"] > 0
    r = idx.build(cfg, quiet=True)
    assert r["indexed"] == 0
    (cfg.roots[0] / "home" / "garden.md").write_text("# Garden\nchanged\n")
    (cfg.roots[0] / "projects" / "journal-6.md").unlink()
    r = idx.build(cfg, quiet=True)
    assert r["indexed"] == 1 and r["removed"] == 1 and r["total"] == 8


def test_search_returns_paths(cfg):
    res = search(cfg, "fusion constant", 5)
    assert res and "projects/search-notes.md" in [p for p, _ in res]


def test_rare_leg(cfg):
    rare = RareIndex([(d, a) for d, a, _ in idx.iter_files(cfg)], cfg.roots)
    assert rare.leg("what broke in ingest-sync-daemon?")[0] == "projects/search-notes.md"


def test_recall_reranks_to_answer(cfg):
    cfg.reranker = {"backend": "fake"}
    r = Recaller(cfg, KeywordFake()).recall("Which electricity plan did we switch to?", top=3, k=9)
    assert r.hits[0].path == "home/utilities.md"
    assert r.hits[0].score == pytest.approx(0.95)
    assert "Saver Plus 12" in r.hits[0].passage
    assert r.usd > 0 and cfg.ledger.exists()


def test_recall_json_reports_tokens(cfg, capsys):
    import deeprecall.cli as cli
    cfg.reranker = {"backend": "fake"}
    r = Recaller(cfg, KeywordFake()).recall("Which electricity plan did we switch to?", top=3, k=9)
    cli._print_recall(r, True, False)
    out = json.loads(capsys.readouterr().out)
    assert out["tokens"] == r.tokens > 0


def test_recall_widens_when_unsure(cfg):
    class Narrow(Recaller):          # first stage misses at k=1, finds it when widened
        def first_stage(self, q, k):
            return ["home/garden.md"] if k == 1 else ["home/garden.md", "home/utilities.md"]
    cfg.reranker = {"backend": "fake"}
    cfg.widen_k = 50
    r = Narrow(cfg, KeywordFake()).recall("Which electricity plan did we switch to?", top=3, k=1)
    assert r.widened and r.hits[0].path == "home/utilities.md"


def test_keyword_query_skips_reranker(cfg):
    cfg.reranker = {"backend": "fake"}
    r = Recaller(cfg, KeywordFake()).recall("electricity plan", top=3)
    assert r.mode == "keyword" and r.hits[0].score is None and r.usd == 0


def test_fail_open(cfg):
    class Boom(KeywordFake):
        def score(self, q, p):
            raise RuntimeError("HTTP 402")
    cfg.reranker = {"backend": "boom"}
    r = Recaller(cfg, Boom()).recall("Which electricity plan did we switch to?")
    assert r.mode.startswith("first-stage") and r.hits and "402" in r.note


def test_budget_caps(tmp_path):
    b = Budget(tmp_path / "l.jsonl", 0.01, 0.02)
    b.check(0.005)
    with pytest.raises(BudgetExceeded):
        b.check(0.02)
    b.record("x", 1000, 0.015)
    b2 = Budget(tmp_path / "l.jsonl", 0.05, 0.02)
    with pytest.raises(BudgetExceeded):
        b2.check(0.01)


def test_budget_refusal_falls_back(cfg):
    cfg.reranker = {"backend": "fake"}
    cfg.max_usd_per_query = 1e-9
    r = Recaller(cfg, KeywordFake()).recall("Which electricity plan did we switch to?")
    assert r.mode.startswith("first-stage") and "BudgetExceeded" in r.note


def test_command_backend(cfg, tmp_path):
    script = tmp_path / "scorer.py"
    script.write_text("import json,sys\nd=json.load(sys.stdin)\n"
                      "print(json.dumps([0.9 if 'Saver Plus' in p['text'] else 0.2 for p in d['passages']]))\n")
    cfg.reranker = {"backend": "command", "command": [sys.executable, str(script)], "window_words": 60}
    r = Recaller(cfg).recall("Which electricity plan did we switch to?", k=9)
    assert r.hits[0].path == "home/utilities.md" and r.hits[0].score == pytest.approx(0.9)


def test_custom_backend_by_import_path():
    rr = load_reranker({"backend": "deeprecall.rerankers.base:NoReranker"})
    assert rr.score("q", [Passage("a", "", "x"), Passage("b", "", "y")]) == [1.0, 0.5]


def test_unknown_backend_message():
    with pytest.raises(Exception) as e:
        load_reranker({"backend": "nope"})
    assert "available" in str(e.value)


def test_prefilter_keeps_matching_windows():
    from deeprecall.recall import prefilter
    wins = [Passage("a.md", str(i), "lorem ipsum filler text") for i in range(10)]
    wins.append(Passage("a.md", "hit", "we switched the electricity plan to Saver Plus"))
    wins.append(Passage("b.md", "only", "short file"))
    kept = prefilter("Which electricity plan did we switch to?", wins, 2)
    assert any(w.heading == "hit" for w in kept)
    assert len([w for w in kept if w.path == "a.md"]) == 2 and any(w.path == "b.md" for w in kept)


def test_jev_key_from_file(tmp_path):
    from deeprecall.rerankers.jev import _key_from_file
    f = tmp_path / "k.env"
    f.write_text("# c\nTYPESAFE_API_KEY=abc123\n")
    assert _key_from_file(str(f), "TYPESAFE_API_KEY") == "abc123"
    f.write_text("rawkey\n")
    assert _key_from_file(str(f), "TYPESAFE_API_KEY") == "rawkey"


def test_log_has_first_stage_and_report(cfg, capsys):
    cfg.reranker = {"backend": "fake"}
    Recaller(cfg, KeywordFake()).recall("Which electricity plan did we switch to?", k=9)
    Recaller(cfg, KeywordFake()).recall("electricity plan")
    rows = [json.loads(l) for l in open(cfg.recall_log)]
    assert rows[0]["first_stage_top3"] and rows[1]["mode"] == "keyword"
    import deeprecall.cli as cli
    import os
    os.environ["DEEPRECALL_CONFIG"] = str(cfg.source)
    try:
        cli.main(["report", "--days", "1"])
    finally:
        del os.environ["DEEPRECALL_CONFIG"]
    rep = json.loads(capsys.readouterr().out)
    assert rep["queries"] == 2 and rep["reranked"] == 1 and rep["confident_top1"] == 1


def test_concurrent_build_is_skipped(cfg):
    import fcntl
    with open(str(cfg.index) + ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        r = idx.build(cfg, quiet=True)
    assert r.get("skipped")
    assert idx.build(cfg, quiet=True).get("skipped") is None
