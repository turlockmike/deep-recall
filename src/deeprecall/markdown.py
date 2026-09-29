"""Markdown parsing: frontmatter, heading-bounded section units, reranker windows."""
from __future__ import annotations

import re

import yaml

_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_H = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def split_frontmatter(text: str) -> tuple[dict, str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    return fm, text[m.end():]


def title_of(path: str, fm: dict, body: str) -> str:
    t = fm.get("title")
    if t:
        return str(t)
    m = re.search(r"^#\s+(.+)$", body, re.M)
    return m.group(1).strip() if m else path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def split_units(body: str, max_words: int = 350) -> list[tuple[str, int, str]]:
    """Heading-bounded units of <= max_words words: [(heading_path, start_word, text)].
    Headings inside ``` fences are ignored. Long sections are split into consecutive pieces."""
    units: list[tuple[str, int, str]] = []
    stack: list[tuple[int, str]] = []
    cur: list[str] = []
    cur_h, start, wpos, in_code = "", 0, 0, False

    def flush():
        txt = "\n".join(cur).strip()
        if not txt:
            return
        w = txt.split()
        if len(w) <= max_words:
            units.append((cur_h, start, txt))
        else:
            for i in range(0, len(w), max_words):
                units.append((cur_h, start + i, " ".join(w[i:i + max_words])))

    for ln in body.split("\n"):
        if ln.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _H.match(ln)
        if m:
            flush()
            cur = []
            lvl = len(m.group(1))
            stack = [s for s in stack if s[0] < lvl] + [(lvl, m.group(2))]
            cur_h = " > ".join(s[1] for s in stack)
            start = wpos
        cur.append(ln)
        wpos += len(ln.split())
    flush()
    return units


def windows(body: str, words: int = 600, max_windows: int = 24) -> list[tuple[str, str]]:
    """Pack consecutive heading units into reranker windows: [(heading_path, text)].
    Window size is widened so one file yields at most ~max_windows windows."""
    total = len(body.split())
    size = max(words, total // max(max_windows, 1) + 1)
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    n, head = 0, ""
    for hp, _sw, tx in split_units(body, min(350, size)):
        k = len(tx.split())
        if cur and n + k > size:
            out.append((head, "\n\n".join(cur)))
            cur, n = [], 0
        if not cur:
            head = hp
        cur.append(f"[{hp}]\n{tx}" if hp else tx)
        n += k
    if cur:
        out.append((head, "\n\n".join(cur)))
    return out or [("", body[:20000])]
