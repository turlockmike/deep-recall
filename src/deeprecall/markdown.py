"""Markdown parsing: frontmatter, heading-bounded section units, reranker windows."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

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


class _HTMLText(HTMLParser):
    """HTML -> Markdown-ish text: <h1>-<h6> become '#' headings (so section splitting works),
    block tags become line breaks, <script>/<style>/<svg>/<template> are dropped."""
    SKIP = {"script", "style", "svg", "template", "noscript", "head"}
    BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "table", "ul", "ol", "pre", "blockquote", "header", "footer", "main", "nav", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self.skip += 1
        elif not self.skip and re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif not self.skip and tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif not self.skip and (re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK):
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    try:
        p.feed(html)
        p.close()
    except Exception:  # malformed HTML: index whatever was extracted so far
        pass
    body = re.sub(r"[ \t]+", " ", "".join(p.out))
    body = re.sub(r"\n\s*\n\s*\n+", "\n\n", body).strip()
    title = " ".join(p.title.split())
    if title and not body.lstrip().startswith("# "):
        body = f"# {title}\n\n{body}"
    return body


def read_doc(path: Path) -> str:
    """Read an indexable file as Markdown text (HTML is converted; everything else is read as-is)."""
    text = path.read_text(errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        return html_to_text(text)
    return text
