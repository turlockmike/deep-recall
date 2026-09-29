"""Rare-term leg: exact-match distinctive query tokens (tool names, IDs, tickers, ALLCAPS,
capitalised phrases, quoted strings) and keep files where the term is rare (df <= max_df).

Embeddings blur identifiers and BM25 tokenisers split them; an exact match on a rare token is
the cheapest recall fix there is. Uses ripgrep when available, else a pure-Python scan.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

STOP = {"What", "Which", "When", "Where", "Why", "How", "Who", "Does", "Did", "Is", "Was", "The", "In", "On",
        "For", "After", "Before", "At", "A", "An", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December", "I", "My", "We", "Our"}


def terms(q: str) -> list[str]:
    t = set()
    for m in re.finditer(r"[A-Za-z0-9][\w./-]*[\w]", q):
        w = m.group(0)
        if len(w) < 3 or (len(w) == 3 and not w.isupper()):
            continue
        if ("-" in w or "_" in w or "/" in w or "." in w.strip(".")
                or re.search(r"[a-z][A-Z]", w) or (w.isupper() and len(w) >= 3)
                or (re.search(r"\d", w) and re.search(r"[A-Za-z]", w))):
            t.add(w)
    for c in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", q):
        ws = [w for w in c.split() if w not in STOP]
        if len(ws) >= 2:
            t.add(" ".join(ws))
    for c in re.findall(r"[\"`']([^\"`']{4,60})[\"`']", q):
        t.add(c)
    return sorted(t)


class RareIndex:
    def __init__(self, files: list[tuple[str, Path]], roots: list[Path]):
        """files: [(display_path, absolute_path)] -- only these are candidates (excludes respected)."""
        self.by_abs = {str(a.resolve()): d for d, a in files}
        self.files = files
        self.roots = [str(r) for r in roots]
        self.rg = shutil.which("rg")
        self._text: dict[str, str] | None = None
        self._cache: dict[str, tuple[str, ...]] = {}

    def files_for(self, term: str) -> tuple[str, ...]:
        if term in self._cache:
            return self._cache[term]
        if self.rg:
            r = subprocess.run([self.rg, "-l", "-i", "-F", term, "--glob", "*.md", *self.roots],
                               capture_output=True, text=True, timeout=60)
            hits = tuple(self.by_abs[k] for k in (str(Path(p).resolve()) for p in r.stdout.splitlines())
                         if k in self.by_abs)
        else:
            if self._text is None:
                self._text = {d: a.read_text(errors="replace").lower() for d, a in self.files}
            t = term.lower()
            hits = tuple(d for d, txt in self._text.items() if t in txt)
        self._cache[term] = hits
        return hits

    def leg(self, q: str, max_df: int = 12) -> list[str]:
        score: dict[str, float] = {}
        ts = terms(q)
        for t in ts:
            fs = self.files_for(t)
            if 0 < len(fs) <= max_df:
                for f in fs:
                    score[f] = score.get(f, 0.0) + 1.0 / len(fs)
        common = [t for t in ts if len(self.files_for(t)) > max_df]
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                both = set(self.files_for(common[i])) & set(self.files_for(common[j]))
                if 0 < len(both) <= max_df:
                    for f in both:
                        score[f] = score.get(f, 0.0) + 0.5 / len(both)
        return sorted(score, key=lambda f: -score[f])
