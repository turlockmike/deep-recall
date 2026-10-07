"""Offline tests: hash embedder + fake rerankers, no model downloads or network."""
from __future__ import annotations

import json
import shutil
import sys
import time
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


@pytest.mark.xfail(strict=True, reason="code default keyword_weight=0.5 buries keyword-only hits; "
                   "live config runs 1.0 since 2026-10-05. Flip config.py L42+L100 to 1.0 and drop this marker.")
def test_exact_identifier_reaches_rerank_pool():
    """Regression (2026-10-05 fresh_canary root cause): an exact identifier
    (commit hash, session id, name) that ranks #1 on the keyword leg but is
    absent from the vector leg's pool must still fuse into the reranker's
    first-stage k. At w=0.5, 0.5/61 loses to every vector top-50 score -> rank 51."""
    from deeprecall.config import Config
    c = Config()
    vec = [f"v{i}" for i in range(c.vector_pool)]
    fused = [p for p, _ in rrf([(vec, 1.0), (["exact-hit"], c.keyword_weight)], c.rrf_k)]
    assert fused.index("exact-hit") < c.k


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


def test_eval_spend_cannot_starve_live(tmp_path):
    """2026-10-01: one eval spent $1.02 of the shared cap and live recall ran unreranked ~24h."""
    led = tmp_path / "l.jsonl"
    ev = Budget(led, 5.0, 2.0, caller="eval")
    for _ in range(3):
        ev.record("jev", 1000, 0.50)  # eval-tagged ledger at $1.50
    live = Budget(led, 0.05, 1.0)
    live.check(0.04)  # live cap $1.00 untouched by $1.50 of eval rows
    assert live.day_usd() == 0.0 and ev.day_usd() == 1.5
    with pytest.raises(BudgetExceeded):
        Budget(led, 5.0, 1.0, caller="eval").check(0.01)  # eval still hits ITS OWN cap
    with open(led, "a") as f:  # legacy untagged rows count as LIVE (never under-count the live cap)
        f.write(json.dumps({"ts": time.time(), "backend": "jev", "tokens": 1, "usd": 0.99}) + "\n")
    with pytest.raises(BudgetExceeded):
        live.check(0.04)
    assert json.loads(led.read_text().splitlines()[0])["caller"] == "eval"


def test_eval_recaller_tags_eval(cfg):
    cfg.reranker = {"backend": "fake"}
    rc = Recaller(cfg, KeywordFake())
    rc.caller = "eval"
    cfg.eval_daily_cap_usd, cfg.daily_cap_usd = 1e-9, 100.0
    r = rc.recall("Which electricity plan did we switch to?")
    assert r.mode.startswith("first-stage") and "eval cap" in r.note, (r.mode, r.note)  # eval hits ITS cap
    live = Recaller(cfg, KeywordFake()).recall("Which electricity plan did we switch to?")
    assert live.mode.startswith("rerank"), (live.mode, live.note)  # live unaffected by the tiny eval cap
    rows = [json.loads(l) for l in cfg.ledger.read_text().splitlines()]
    assert rows and {r["caller"] for r in rows} == {"live"}


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


def test_cli_index_lock_skip_exits_tempfail(cfg, monkeypatch, capsys):
    """A lock-skipped `deeprecall index` did no work, so it must not exit 0 like a real run."""
    import fcntl
    from deeprecall import cli
    monkeypatch.setenv("DEEPRECALL_CONFIG", str(cfg.source))
    with open(str(cfg.index) + ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert cli.main(["index", "--quiet"]) == 75
    assert "locked" in capsys.readouterr().err
    assert cli.main(["index", "--quiet"]) == 0


class KindFake(KeywordFake):
    """KeywordFake that classifies every question as `kind`, and scores `marker_for` kinds by another marker."""

    def __init__(self, kind="aggregate", marker="Saver Plus 12", marker_for=None, **_):
        super().__init__(marker)
        self.kind, self.marker_for, self.classified = kind, marker_for or {}, 0

    def classify(self, q):
        self.classified += 1
        self.last_tokens = 40
        if isinstance(self.kind, Exception):
            raise self.kind
        return self.kind

    def for_kind(self, kind):
        return KindFake(self.kind, self.marker_for[kind]) if kind in self.marker_for else self


class OneFileFirst(Recaller):          # the answer is already in a k=1 pool; widening adds the garden note
    def first_stage(self, q, k):
        return ["home/utilities.md"] if k == 1 else ["home/utilities.md", "home/garden.md"]


def test_question_kind_widens_a_confident_pool(cfg):
    cfg.reranker, cfg.widen_k = {"backend": "fake"}, 50
    q = "Which electricity plan did we switch to?"
    assert not OneFileFirst(cfg, KindFake()).recall(q, top=3, k=1).widened     # question_kinds is off
    cfg.question_kinds = True
    r = OneFileFirst(cfg, KindFake("aggregate")).recall(q, top=3, k=1)
    assert r.kind == "aggregate" and r.widened and r.hits[0].path == "home/utilities.md"
    assert not OneFileFirst(cfg, KindFake("single_fact")).recall(q, top=3, k=1).widened


def test_question_kind_picks_the_reranker(cfg):
    cfg.reranker, cfg.question_kinds = {"backend": "fake"}, True
    q = "Which electricity plan did we switch to?"
    rr = KindFake("preference", marker_for={"preference": "Cherokee Purple"})
    r = Recaller(cfg, rr).recall(q, top=3, k=9)
    assert r.kind == "preference" and r.hits[0].path == "home/garden.md"
    assert Recaller(cfg, KindFake("single_fact", marker_for={"preference": "Cherokee Purple"})).recall(q, top=3, k=9) \
        .hits[0].path == "home/utilities.md"


def test_failed_classification_only_loses_the_kind(cfg):
    cfg.reranker, cfg.question_kinds = {"backend": "fake"}, True
    r = Recaller(cfg, KindFake(RuntimeError("HTTP 500"))).recall("Which electricity plan did we switch to?", k=9)
    assert r.mode.startswith("rerank") and r.kind is None and r.hits[0].path == "home/utilities.md"


def test_classification_is_billed_and_logged(cfg):
    cfg.reranker, cfg.question_kinds = {"backend": "fake"}, True
    rr = KindFake("temporal")
    r = Recaller(cfg, rr).recall("Which electricity plan did we switch to?", k=9)
    assert rr.classified == 1 and r.tokens > 40
    log = [json.loads(l) for l in open(cfg.recall_log)]
    assert log[-1]["kind"] == "temporal"


def test_question_kinds_config(tmp_path):
    c = tmp_path / "config.toml"
    c.write_text('roots = ["."]\n[recall]\nquestion_kinds = true\nwiden_kinds = ["aggregate"]\n')
    conf = load(c)
    assert conf.question_kinds and conf.widen_kinds == ["aggregate"]
    c.write_text('roots = ["."]\n')
    assert not load(c).question_kinds and load(c).widen_kinds == ["aggregate", "temporal"]


def test_jev_classify_and_kind_prompts(monkeypatch):
    from deeprecall.rerankers import jev
    sent = []

    def fake_post(url, body, headers, timeout=60, **_):
        sent.append(body)
        return {"answers": {"kind": {"choice": reply[0]}}, "usage": {"input_tokens": 123}}
    monkeypatch.setattr(jev, "post_json", fake_post)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    reply = ["preference"]
    rr = jev.JevReranker()
    assert rr.classify("Any tips for my commute?") == "preference" and rr.last_tokens == 123
    q = sent[-1]["questions"]["kind"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(jev.KINDS)
    assert sent[-1]["state"] == {"question": "Any tips for my commute?"}
    reply[0] = "something_else"
    assert rr.classify("?") is None
    tuned = rr.for_kind("preference")
    assert tuned.prompt == jev.KIND_PROMPTS["preference"] and rr.prompt == jev.PROMPT
    assert rr.for_kind("aggregate") is rr and rr.for_kind(None) is rr
    assert jev.JevReranker(kind_prompts={}).for_kind("preference").prompt == jev.PROMPT


def test_widen_that_adds_nothing_is_not_billed_again(cfg):
    class SamePool(Recaller):          # widening finds no file the first pass didn't already score
        def first_stage(self, q, k):
            return ["home/utilities.md", "home/garden.md"]
    cfg.reranker, cfg.question_kinds, cfg.widen_k = {"backend": "fake"}, True, 50
    rr = KindFake("aggregate")
    r = SamePool(cfg, rr).recall("Which electricity plan did we switch to?", top=3, k=2)
    first_pass = sum(1 for _ in open(cfg.ledger))
    assert not r.widened
    ledger = [json.loads(l) for l in open(cfg.ledger)]
    assert first_pass == 2 and r.tokens == 40 + ledger[-1]["tokens"]


def test_classify_tokens_are_recorded_when_the_reply_is_malformed(cfg):
    class Malformed(KindFake):
        def classify(self, q):
            self.last_tokens = 40
            raise KeyError("kind")
    cfg.reranker, cfg.question_kinds = {"backend": "fake"}, True
    r = Recaller(cfg, Malformed()).recall("Which electricity plan did we switch to?", k=9)
    ledger = [json.loads(l) for l in open(cfg.ledger)]
    assert r.kind is None and ledger[0]["tokens"] == 40 and r.tokens > 40


def test_widen_kinds_accepts_a_bare_string(tmp_path):
    c = tmp_path / "config.toml"
    c.write_text('roots = ["."]\n[recall]\nwiden_kinds = "aggregate"\n')
    assert load(c).widen_kinds == ["aggregate"]


def test_jev_classify_survives_a_reply_without_answers(monkeypatch):
    from deeprecall.rerankers import jev
    monkeypatch.setattr(jev, "post_json", lambda *a, **k: {"usage": {"input_tokens": 77}})
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    rr = jev.JevReranker()
    assert rr.classify("How many?") is None and rr.last_tokens == 77


def test_file_root_is_indexed_and_resolved(tmp_path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\nalpha note\n")
    single = tmp_path / "journal.md"; single.write_text("# Journal\nMike switched the electric plan\n")
    (tmp_path / "data.txt").write_text("not markdown")
    c = tmp_path / "config.toml"
    c.write_text(f'roots = ["{notes}", "{single}", "{tmp_path / "data.txt"}"]\nindex = "{tmp_path / "state" / "index.db"}"\n'
                 '[embedding]\nmodel = "hash"\n')
    cfg = load(c)
    disps = {d for d, _a, _r in idx.iter_files(cfg)}
    assert disps == {"notes/a.md", "journal.md"}
    assert idx.resolve(cfg, "journal.md") == single
    assert idx.resolve(cfg, "notes/a.md") == notes / "a.md"
    one = load(c) ; one.roots = [single]
    assert {d for d, _a, _r in idx.iter_files(one)} == {"journal.md"} and idx.resolve(one, "journal.md") == single


def test_html_extension_indexed_as_text(tmp_path):
    from deeprecall.markdown import html_to_text, read_doc
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# A\nalpha\n")
    (notes / "p.html").write_text("<html><head><title>Keel proposal</title><style>.x{color:red}</style></head><body>"
                                  "<h2>Decision 1</h2><p>Mike picks the fact table.</p>"
                                  "<script>var secretjs = 1;</script></body></html>")
    (notes / "skip.txt").write_text("not indexed")
    c = tmp_path / "config.toml"
    c.write_text(f'roots = ["{notes}"]\nindex = "{tmp_path / "state" / "index.db"}"\nextensions = ["md", ".html"]\n'
                 '[embedding]\nmodel = "hash:64"\n[reranker]\nbackend = "none"\n')
    cfg = load(c)
    assert cfg.extensions == [".md", ".html"]
    assert {d for d, _a, _r in idx.iter_files(cfg)} == {"a.md", "p.html"}
    t = read_doc(notes / "p.html")
    assert "## Decision 1" in t and "fact table" in t and "secretjs" not in t and "color:red" not in t
    assert t.startswith("# Keel proposal")
    idx.build(cfg, quiet=True)
    assert "p.html" in [r[0] for r in search(cfg, "fact table decision")]
    default = load(c); default.extensions = [".md"]
    assert {d for d, _a, _r in idx.iter_files(default)} == {"a.md"}
    assert html_to_text("<p>unclosed <b>tag") .startswith("unclosed")


def test_eval_cap_zero_refuses_live_cap_zero_unlimited(tmp_path):
    # Mike 2026-10-01 20:03: eval_daily_cap_usd = 0 must REFUSE evals (not mean "unlimited").
    from deeprecall.budget import Budget
    from deeprecall.rerankers.base import BudgetExceeded
    import pytest
    with pytest.raises(BudgetExceeded):
        Budget(tmp_path / "l.jsonl", 0.08, 0.0, "eval").check(0.001)
    Budget(tmp_path / "l.jsonl", 0.08, 0.0, "live").check(0.001)


def test_budget_widen_reserve(tmp_path):
    """2026-10-03: widens ate the $0.60 live cap on 10-02 and later recalls ran unreranked.
    A widen (ceiling < 1) must stop at (1 - reserve) of the cap; a base rerank may still use the reserve."""
    import json, time
    from deeprecall.budget import Budget
    from deeprecall.rerankers.base import BudgetExceeded
    led = tmp_path / "spend.jsonl"
    led.write_text(json.dumps({"ts": time.time(), "usd": 0.38, "caller": "live"}) + "\n")
    b = Budget(led, 0.08, 0.60, "live")
    b.ceiling = 0.65                        # widen pass: 0.38 + 0.05 > 0.39 -> refused
    try:
        b.check(0.05)
        assert False, "widen should hit the reserve"
    except BudgetExceeded as e:
        assert "widen reserve" in str(e)
    b.ceiling = 1.0                         # base pass: 0.38 + 0.01 <= 0.60 -> allowed
    b.check(0.01)


def test_budget_session_share(tmp_path):
    """2026-10-06: one session's 49-recall burst ate 61% of the $0.60 cap on 10-05 and starved tg-delegate.
    One session may fill at most session_share_frac of the cap; other sessions and session-less rows are unaffected."""
    import json, time
    from deeprecall.budget import Budget
    from deeprecall.rerankers.base import BudgetExceeded
    led = tmp_path / "spend.jsonl"
    now = time.time()
    led.write_text("".join(json.dumps(r) + "\n" for r in [
        {"ts": now, "usd": 0.23, "caller": "live", "session": "burst"},
        {"ts": now, "usd": 0.10, "caller": "live"},                       # legacy/no session
        {"ts": now - 90000, "usd": 0.50, "caller": "live", "session": "burst"}]))  # outside 24h
    b = Budget(led, 0.08, 0.60, "live", session="burst", session_share_frac=0.4)
    with pytest.raises(BudgetExceeded, match="session share"):
        b.check(0.02)                        # 0.23 + 0.02 > 0.24
    b.check(0.005)                           # 0.235 <= 0.24
    Budget(led, 0.08, 0.60, "live", session="other", session_share_frac=0.4).check(0.05)
    Budget(led, 0.08, 0.60, "live", session=None, session_share_frac=0.4).check(0.05)
    Budget(led, 0.08, 0.60, "live", session="burst", session_share_frac=0.0).check(0.05)   # 0 = off
    b.record("jev", 10, 0.001)
    assert json.loads(led.read_text().splitlines()[-1])["session"] == "burst"


def test_jev_local_primary_falls_back_sticky(monkeypatch):
    """base_url = local jev-compatible server ($0); transport failure -> fallback for every copy, priced."""
    from deeprecall.rerankers import jev
    from deeprecall.rerankers.base import Passage, RerankerError
    calls = []

    def fake_post(url, body, headers, timeout=60, **_):
        calls.append((url, body["model"]))
        if url.startswith("http://tower"):
            raise RerankerError("<urlopen error timed out>")
        return {"answers": {"q": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10}}
    monkeypatch.setattr(jev, "post_json", fake_post)
    monkeypatch.setattr(jev, "_reachable", lambda url, timeout=1.5: True)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    rr = jev.JevReranker(base_url="http://tower:1/v1", model="clef-flash:latest", usd_per_mtok=0, workers=4,
                         fallback_base_url="https://api.typesafe.ai/v1")
    assert rr.usd_per_token == 0 and rr.workers == 4
    tuned = rr.for_kind("preference")
    assert tuned.score("q?", [Passage("a.md", "x", "")]) == [0.8]
    assert calls == [("http://tower:1/v1/systemone", "clef-flash:latest"),
                     ("https://api.typesafe.ai/v1/systemone", "jev-latest")]
    assert rr._route["fell_back"] and rr.usd_per_token == 0.042 / 1e6 and rr.workers == 32   # shared by copies
    rr.score("q?", [Passage("b.md", "y", "")])
    assert calls[-1][0].startswith("https://api.typesafe.ai")                               # no tower retry

    monkeypatch.setattr(jev, "_reachable", lambda url, timeout=1.5: False)                  # down at startup
    assert jev.JevReranker(base_url="http://tower:1/v1", fallback_base_url="https://x/v1").url == "https://x/v1/systemone"


# --- EmbeddingGemma 2 embedder (remote-first; fake local HTTP server, no model) ---------------------------------
def _fake_embed_server(dim_default=768):
    import json as _json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["content-length"])))
            calls.append(body)
            dim = body.get("dimensions", dim_default)
            data = []
            for i, t in enumerate(body["input"]):
                v = [0.0] * dim
                v[i % dim] = 2.0          # deliberately NOT unit: the client must renormalize
                v[(i + 1) % dim] = 1.0
                data.append({"index": i, "embedding": v})
            out = _json.dumps({"data": data[::-1], "usage": {"prompt_tokens": 1}}).encode()   # out of order on purpose
            self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def test_eg2_prefixes_and_remote_batching():
    from deeprecall.embed import Eg2Embedder, get_embedder
    assert Eg2Embedder.doc_text("Title\nbody") == "title: Title | text: body"
    assert Eg2Embedder.doc_text("T > h1 > h2\nsec") == "title: T > h1 > h2 | text: sec"
    assert Eg2Embedder.doc_text("bare") == "title: none | text: bare"
    assert Eg2Embedder.query_text("why?") == "task: search result | query: why?"
    srv, calls = _fake_embed_server()
    url = f"http://127.0.0.1:{srv.server_port}/v1/embeddings"
    e = get_embedder("google/embeddinggemma-2", {"remote_url": url, "batch_size": 2, "dim": 768})
    texts = [f"T{i}\n" + "x" * i for i in (5, 1, 9, 3, 7)]           # unsorted lengths
    vs = e.embed(texts)
    assert len(vs) == 5 and all(len(v) == 768 for v in vs)
    for v in vs:                                                     # renormalized + order restored
        assert abs(sum(x * x for x in v) - 1.0) < 1e-6
    # batches are length-sorted (the server pads each batch to its longest text); calls land in completion order
    batches = sorted((c["input"] for c in calls), key=lambda b: len(b[0]))
    sent = [t for b in batches for t in b]
    assert sorted(sent, key=len) == sent and len(sent) == 5 and len(calls) == 3
    assert all(t.startswith("title: T") for t in sent)
    q = e.embed_query(["q?"])[0]
    assert calls[-1]["input"] == ["task: search result | query: q?"] and "dimensions" not in calls[-1]
    e256 = get_embedder("google/embeddinggemma-2", {"remote_url": url, "dim": 256})
    assert len(e256.embed(["a\nb"])[0]) == 256 and calls[-1]["dimensions"] == 256
    assert "model" not in calls[-1]                                  # desktop server: no model field
    eo = get_embedder("google/embeddinggemma-2", {"remote_url": url, "remote_model": "embeddinggemma-2:270m"})
    eo.embed_query(["q?"])                                           # Ollama /v1/embeddings needs "model"
    assert calls[-1]["model"] == "embeddinggemma-2:270m"
    srv.shutdown()


def test_eg2_build_refuses_cpu_fallback_when_remote_down(tmp_path):
    from deeprecall.embed import Eg2Embedder
    e = Eg2Embedder({"remote_url": "http://127.0.0.1:9/v1/embeddings", "remote_timeout": 0.5, "query_timeout": 0.5,
                     "local_onnx": "/nonexistent", "down_marker": str(tmp_path / "down"), "slot_dir": str(tmp_path)})
    from deeprecall.embed import RemoteDown
    with pytest.raises(RemoteDown, match="refused"):                  # index.py turns this into FTS-first mode
        e.embed(["T\nx"])
    with pytest.raises(FileNotFoundError):                            # queries DO fall back (model missing here)
        e.embed_query(["q"])
    assert e.stats["fallbacks"] == 1


def test_eg2_query_fallback_is_fast_and_sticky(tmp_path):
    """Tower off: one failed query marks the remote down (shared marker file), later queries skip it; no retries.
    2026-10-06: without this, every search paid 2 retries + sleeps = 14 s instead of ~0.3 s int8."""
    from deeprecall.embed import Eg2Embedder
    calls = {"n": 0}
    mk = str(tmp_path / "down")
    e = Eg2Embedder({"remote_url": "http://127.0.0.1:9/v1/embeddings", "down_marker": mk})
    def boom(texts, timeout):
        calls["n"] += 1
        raise OSError("refused")
    e._post = boom
    e._local_embed = lambda texts: [[1.0] for _ in texts]
    t0 = __import__("time").time()
    assert e.embed_query(["q"]) == [[1.0]] and calls["n"] == 1        # retries=0 for queries
    assert __import__("os").path.exists(mk)
    e2 = Eg2Embedder({"remote_url": "http://127.0.0.1:9/v1/embeddings", "down_marker": mk})   # new process, same marker
    e2._post = boom; e2._local_embed = lambda texts: [[2.0] for _ in texts]
    assert e2.embed_query(["q"]) == [[2.0]] and calls["n"] == 1      # remote skipped while marked down
    assert __import__("time").time() - t0 < 1.0


def test_eg2_down_marker_is_per_url():
    from deeprecall.embed import Eg2Embedder
    a = Eg2Embedder({"remote_url": "http://127.0.0.1:9/v1/embeddings"})
    b = Eg2Embedder({"remote_url": "http://192.168.1.78:8089/v1/embeddings"})
    assert a.down_marker != b.down_marker


def test_eg2_build_raises_remote_down_not_systemexit(tmp_path):
    from deeprecall.embed import Eg2Embedder, RemoteDown
    e = Eg2Embedder({"remote_url": "http://127.0.0.1:9/v1/embeddings", "remote_timeout": 0.5, "local_onnx": "/nonexistent",
                     "down_marker": str(tmp_path / "down"), "slot_dir": str(tmp_path)})
    with pytest.raises(RemoteDown, match="refused"):
        e.embed(["T\nx"])


def test_index_fts_first_when_remote_down(tmp_path, monkeypatch):
    """Tower off: the build still writes docs + FTS (keyword-findable), marks them pending, exits normally; the next
    run with the remote up embeds exactly the pending docs (hash unchanged) and clears the flag."""
    import deeprecall.index as idx
    from deeprecall import embed as em
    from deeprecall.search import keyword_leg
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "a.md").write_text("# Alpha\n\nzebra quartz lantern\n")
    (notes / "b.md").write_text("# Beta\n\nplain words here\n")
    from deeprecall.config import Config
    cfg = Config(roots=[notes], index=tmp_path / "i.db", embed_model="hash:16")
    state = {"down": True}

    class Flaky(em.HashEmbedder):
        def embed(self, texts):
            if state["down"]:
                raise em.RemoteDown("remote refused")
            return super().embed(texts)
    monkeypatch.setattr(idx, "get_embedder", lambda model, opts=None: Flaky(16))

    r = idx.build(cfg)
    assert r["indexed"] == 2 and r["pending"] == 2 and "remote_down" in r
    db = idx.connect(cfg)
    assert db.execute("SELECT COUNT(*) FROM docs_vec").fetchone()[0] == 0
    assert [p for p in keyword_leg(db, "zebra quartz", 5)] == ["a.md"]       # keyword-findable while pending
    db.close()
    assert idx.stats(cfg)["pending_embed"] == 2

    state["down"] = False
    r2 = idx.build(cfg)                                                       # remote back: only the pending docs redo
    assert r2["indexed"] == 2 and r2["pending"] == 0 and "remote_down" not in r2
    db = idx.connect(cfg)
    assert db.execute("SELECT COUNT(*) FROM docs_vec").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM docs WHERE embedded=0").fetchone()[0] == 0
    db.close()
    r3 = idx.build(cfg)
    assert r3["indexed"] == 0                                                 # nothing left to do

    # remote down again, a.md is EDITED: text/FTS refresh, but its old vector stays (stale vector beats none)
    state["down"] = True
    (notes / "a.md").write_text("# Alpha\n\nzebra quartz lantern plus new words\n")
    (notes / "c.md").write_text("# Gamma\n\nbrand new note\n")
    r4 = idx.build(cfg)
    assert r4["indexed"] == 2 and r4["pending"] == 2
    db = idx.connect(cfg)
    a_id = db.execute("SELECT id FROM docs WHERE path='a.md'").fetchone()[0]
    assert db.execute("SELECT COUNT(*) FROM docs_vec WHERE id=?", (a_id,)).fetchone()[0] == 1   # kept
    assert db.execute("SELECT COUNT(*) FROM docs_vec").fetchone()[0] == 2                        # c.md has none yet
    assert "new words" in db.execute("SELECT body FROM docs_fts WHERE path='a.md'").fetchone()[0]
    db.close()
    state["down"] = False
    r5 = idx.build(cfg)
    assert r5["indexed"] == 2 and r5["pending"] == 0
    db = idx.connect(cfg)
    assert db.execute("SELECT COUNT(*) FROM docs_vec").fetchone()[0] == 3
    db.close()


def test_eg2_local_slots_cap_concurrent_model_loads(tmp_path):
    import fcntl
    from deeprecall.embed import Eg2Embedder
    e = Eg2Embedder({"remote_url": "", "local_slots": 1, "slot_dir": str(tmp_path)})
    e._acquire_slot()
    other = open(tmp_path / "eg2-local-slot-0.lock", "w")
    with pytest.raises(OSError):                                              # the one slot is held by this process
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    e._slot_fd.close()                                                        # process exit releases it
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_eg2_build_fails_fast_when_host_blackholes(monkeypatch):
    """A powered-off tower black-holes TCP (no RST). The build must learn 'down' in seconds, not remote_timeout x tries."""
    import socket, time
    from deeprecall.embed import Eg2Embedder, RemoteDown
    def hang(*a, **k):
        raise socket.timeout("timed out")
    monkeypatch.setattr(socket, "create_connection", hang)
    e = Eg2Embedder({"remote_url": "http://10.255.255.1:8089/v1/embeddings", "remote_timeout": 120})
    t0 = time.time()
    with pytest.raises(RemoteDown):
        e.embed(["T\nx"])
    assert time.time() - t0 < 5
