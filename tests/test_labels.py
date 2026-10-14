"""[labels]: recall shows each hit's document type and date; frontmatter under a leading comment is read."""
from types import SimpleNamespace

from deeprecall import cli
from deeprecall.markdown import split_frontmatter


def test_frontmatter_under_leading_comment():
    fm, body = split_frontmatter("<!--strata\n{}\n-->\n---\nkind: rule\n---\n# t\n")
    assert fm == {"kind": "rule"} and body.startswith("# t")


def test_label_from_frontmatter_then_folder_default(tmp_path):
    (tmp_path / "a.md").write_text("---\nkind: state\nas_of: 2026-10-13\n---\n# a\n")
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "b.md").write_text("# b\n")
    cfg = SimpleNamespace(roots=[tmp_path], labels={"date_fields": ["as_of", "updated"],
                                                    "defaults": {"j/**": "record"}})
    assert cli._label(cfg, "a.md") == ("state", "2026-10-13")
    assert cli._label(cfg, "j/b.md") == ("record", "")
    assert cli._label(SimpleNamespace(roots=[tmp_path], labels={}), "a.md") == ("", "")
