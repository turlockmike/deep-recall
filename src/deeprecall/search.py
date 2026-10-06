"""First-stage hybrid search: FTS5 BM25 + vector KNN (file vectors, best section for long files),
merged with weighted reciprocal rank fusion."""
from __future__ import annotations

import re

from .config import Config
from .embed import get_embedder
from .index import blob, connect

_FTS_KEYWORDS = {"AND", "OR", "NOT", "NEAR"}


def fts_query(q: str) -> str:
    """Natural language -> FTS5-safe OR query of bare alphanumeric tokens (punctuation is FTS5 syntax)."""
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", q) if t not in _FTS_KEYWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in toks)


def keyword_leg(db, q: str, limit: int) -> list[str]:
    fq = fts_query(q)
    if not fq:
        return []
    rows = db.execute("SELECT path FROM docs_fts WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts, 0, 5.0, 1.0) LIMIT ?",
                      (fq, limit)).fetchall()
    return [r[0] for r in rows]


def vector_leg(db, qv: list[float], limit: int, section_pool: int) -> list[str]:
    b = blob(qv)
    best: dict[str, float] = {}
    for path, d in db.execute(
            "SELECT d.path, v.distance FROM docs_vec v JOIN docs d ON d.id = v.id "
            "WHERE v.embedding MATCH ? AND k = ?", (b, limit)):
        best[path] = d
    long_docs = {r[0] for r in db.execute("SELECT DISTINCT path FROM sections")}
    for p in list(best):                    # long docs are scored by their best section, not the diluted mean
        if p in long_docs:
            del best[p]
    for path, d in db.execute(
            "SELECT s.path, v.distance FROM sections_vec v JOIN sections s ON s.sid = v.sid "
            "WHERE v.embedding MATCH ? AND k = ?", (b, section_pool)):
        if d < best.get(path, 9e9):
            best[path] = d
    return [p for p, _ in sorted(best.items(), key=lambda x: x[1])][:limit]


def rrf(lists: list[tuple[list[str], float]], k: int) -> list[tuple[str, float]]:
    score: dict[str, float] = {}
    for ranked, w in lists:
        for i, p in enumerate(ranked):
            score[p] = score.get(p, 0.0) + w / (k + i + 1)
    return sorted(score.items(), key=lambda x: -x[1])


def search(cfg: Config, q: str, top_k: int = 10) -> list[tuple[str, float]]:
    if not cfg.index.exists():
        raise SystemExit(f"no index at {cfg.index}: run `deeprecall index` first")
    emb = get_embedder(cfg.embed_model, cfg.embedding)
    db = connect(cfg)
    try:
        pool = max(top_k, cfg.vector_pool)
        kw = keyword_leg(db, q, pool)
        vec = vector_leg(db, emb.embed_query([q])[0], pool, max(cfg.section_pool, top_k))
    finally:
        db.close()
    fused = rrf([(vec, 1.0), (kw, cfg.keyword_weight)], cfg.rrf_k)
    return fused[:top_k]
