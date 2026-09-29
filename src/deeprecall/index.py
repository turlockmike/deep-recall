"""SQLite index: FTS5 keyword table + sqlite-vec vector tables (file vectors and section vectors).

Incremental: a file is re-indexed only when its content hash changes; deleted files are pruned.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

import sqlite_vec

from .config import Config
from .embed import get_embedder, mean, normalize
from .markdown import split_frontmatter, split_units, title_of


def connect(cfg: Config, dim: int | None = None) -> sqlite3.Connection:
    cfg.index.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cfg.index, timeout=60)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    if dim is not None:
        ensure_schema(db, dim, cfg.embed_model)
    return db


def ensure_schema(db: sqlite3.Connection, dim: int, model: str) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, root TEXT, hash TEXT,
            title TEXT, words INTEGER, indexed_at REAL);
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            path UNINDEXED, title, body, tokenize='porter unicode61');
        CREATE TABLE IF NOT EXISTS sections (
            sid INTEGER PRIMARY KEY, path TEXT NOT NULL, hpath TEXT, start_word INTEGER);
        CREATE INDEX IF NOT EXISTS idx_sections_path ON sections(path);
    """)
    row = db.execute("SELECT v FROM meta WHERE k='embed'").fetchone()
    want = json.dumps({"model": model, "dim": dim})
    if row and row[0] != want:
        raise SystemExit(f"index was built with {row[0]}, config wants {want}: run `deeprecall index --full`")
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])")
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS sections_vec USING vec0(sid INTEGER PRIMARY KEY, embedding float[{dim}])")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('embed', ?)", (want,))
    db.commit()


def blob(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def iter_files(cfg: Config):
    """Yield (display_path, absolute_path, root) for every Markdown file under the roots."""
    multi = len(cfg.roots) > 1
    for root in cfg.roots:
        if not root.is_dir():
            print(f"deeprecall: root not found: {root}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*.md")):
            rel = p.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in cfg.exclude):
                continue
            yield (f"{root.name}/{rel}" if multi else rel), p, root


def resolve(cfg: Config, display_path: str) -> Path:
    """Map an index path back to the file on disk."""
    if len(cfg.roots) == 1:
        return cfg.roots[0] / display_path
    head, _, rest = display_path.partition("/")
    for r in cfg.roots:
        if r.name == head:
            return r / rest
    return cfg.roots[0] / display_path


def _chunks(body: str, n: int) -> list[str]:
    w = body.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def build(cfg: Config, full: bool = False, quiet: bool = False) -> dict:
    emb = get_embedder(cfg.embed_model)
    if full and cfg.index.exists():
        cfg.index.unlink()
        for suf in ("-wal", "-shm"):
            Path(str(cfg.index) + suf).unlink(missing_ok=True)
    db = connect(cfg, emb.dim)
    known = dict(db.execute("SELECT path, hash FROM docs"))
    seen, todo = set(), []
    for disp, absp, root in iter_files(cfg):
        seen.add(disp)
        try:
            text = absp.read_text(errors="replace")
        except OSError:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        if known.get(disp) != h:
            todo.append((disp, absp, root, text, h))
    gone = [p for p in known if p not in seen]
    for p in gone:
        _drop(db, p)
    t0, n, group = time.time(), 0, 32
    for g in range(0, len(todo), group):           # embed many files per model call (much faster)
        batch = []
        for disp, absp, root, text, h in todo[g:g + group]:
            fm, body = split_frontmatter(text)
            title = title_of(disp, fm, body)
            extra = " ".join(str(fm.get(k, "")) for k in ("keywords", "tags", "summary", "description") if fm.get(k))
            summary = str(fm.get("summary") or fm.get("description") or "").strip()
            doc_texts = [f"{title}\n{c}" for c in _chunks(body, cfg.chunk_words)] + ([f"{title}\n{summary}"] if summary else [])
            units = split_units(body, cfg.chunk_words) if len(body.split()) > cfg.chunk_words else []
            unit_texts = [f"{title} > {hp}\n{tx}" for hp, _s, tx in units]
            batch.append((disp, root, h, title, extra, body, doc_texts, units, unit_texts))
        flat = [t for b in batch for t in b[6] + b[8]]
        vecs = emb.embed(flat) if flat else []
        i = 0
        for disp, root, h, title, extra, body, doc_texts, units, unit_texts in batch:
            dv = vecs[i:i + len(doc_texts)]; i += len(doc_texts)
            uv = vecs[i:i + len(unit_texts)]; i += len(unit_texts)
            _drop(db, disp)
            did = db.execute("INSERT INTO docs(path, root, hash, title, words, indexed_at) VALUES (?,?,?,?,?,?)",
                             (disp, str(root), h, title, len(body.split()), time.time())).lastrowid
            db.execute("INSERT INTO docs_fts(rowid, path, title, body) VALUES (?,?,?,?)",
                       (did, disp, f"{title} {extra}", body))
            db.execute("INSERT INTO docs_vec(id, embedding) VALUES (?, ?)", (did, blob(mean(dv))))
            for (hp, sw, _tx), v in zip(units, uv):
                sid = db.execute("INSERT INTO sections(path, hpath, start_word) VALUES (?,?,?)",
                                 (disp, hp, sw)).lastrowid
                db.execute("INSERT INTO sections_vec(sid, embedding) VALUES (?, ?)", (sid, blob(normalize(v))))
            n += 1
        db.commit()
        if not quiet:
            print(f"  indexed {n}/{len(todo)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    db.commit()
    total = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    db.close()
    return {"indexed": n, "removed": len(gone), "total": total, "secs": round(time.time() - t0, 1)}


def _drop(db: sqlite3.Connection, path: str) -> None:
    row = db.execute("SELECT id FROM docs WHERE path=?", (path,)).fetchone()
    if row:
        db.execute("DELETE FROM docs_vec WHERE id=?", (row[0],))
        db.execute("DELETE FROM docs_fts WHERE rowid=?", (row[0],))
        db.execute("DELETE FROM docs WHERE id=?", (row[0],))
    sids = [r[0] for r in db.execute("SELECT sid FROM sections WHERE path=?", (path,))]
    for s in sids:
        db.execute("DELETE FROM sections_vec WHERE sid=?", (s,))
    db.execute("DELETE FROM sections WHERE path=?", (path,))


def stats(cfg: Config) -> dict:
    if not cfg.index.exists():
        return {"index": str(cfg.index), "exists": False}
    db = sqlite3.connect(cfg.index)
    try:
        docs = db.execute("SELECT COUNT(*), COALESCE(SUM(words),0) FROM docs").fetchone()
        secs = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        meta = dict(db.execute("SELECT k, v FROM meta"))
    finally:
        db.close()
    return {"index": str(cfg.index), "exists": True, "docs": docs[0], "words": docs[1],
            "sections": secs, "embedding": json.loads(meta.get("embed", "{}"))}
