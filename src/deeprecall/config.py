"""Configuration: a TOML file + environment overrides.

Lookup order for the config file (first hit wins):
  1. $DEEPRECALL_CONFIG
  2. ./.deeprecall/config.toml   (project-local, walking up from cwd)
  3. ~/.config/deeprecall/config.toml

Minimal config:

    roots = ["~/notes"]

Everything else has a default. See `deeprecall init` for a commented template.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore

TEMPLATE = '''# deeprecall config
# Folders of Markdown notes to index (recursively). Required.
roots = ["~/notes"]

# Where the SQLite index lives. Default: <this config dir>/index.db
# index = "~/.config/deeprecall/index.db"

# Glob patterns (relative to a root) to skip.
exclude = [".git/**", "node_modules/**", "**/.obsidian/**"]

[embedding]
model = "BAAI/bge-small-en-v1.5"   # any fastembed TextEmbedding model
chunk_words = 350                  # file vector = mean of chunk vectors; also the section-unit size

[search]
rrf_k = 60
keyword_weight = 0.5               # vector leg weight is 1.0
vector_pool = 50
section_pool = 400

[recall]
k = 20                             # first-stage depth fed to the reranker
widen_at = "auto"                  # rerank score below this => widen once; "auto" = the reranker's default
widen_k = 150
rare_max_df = 12                   # rare-term leg: keep terms found in <= N files
max_windows = 24                   # per file; windows widen so big files stay bounded
question_gate = true               # only rerank question-shaped queries
question_kinds = false             # classify each question first (jev only); widen_kinds always widen
widen_kinds = ["aggregate", "temporal"]

[reranker]
# backend: "cross-encoder" (local, free), "jev" (TypeSafe API), "openai" (any OpenAI-compatible
# endpoint with logprobs), "command" (any executable), "none", or a registered plugin name.
backend = "cross-encoder"
# model = "Xenova/ms-marco-MiniLM-L-6-v2"

[budget]
max_usd_per_query = 0.05
daily_cap_usd = 1.00               # rolling 24h, across all processes; 0 disables
'''


def _expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def _str_list(v) -> list[str]:
    """A TOML list of strings, accepting a bare string as a one-item list (list("aggregate") is letters)."""
    return [v] if isinstance(v, str) else list(v)


def find_config_file(start: Path | None = None) -> Path | None:
    env = os.environ.get("DEEPRECALL_CONFIG")
    if env:
        return _expand(env)
    d = (start or Path.cwd()).resolve()
    for p in [d, *d.parents]:
        c = p / ".deeprecall" / "config.toml"
        if c.is_file():
            return c
    home = Path.home() / ".config" / "deeprecall" / "config.toml"
    return home if home.is_file() else None


@dataclass
class Config:
    roots: list[Path]
    index: Path
    exclude: list[str] = field(default_factory=lambda: [".git/**", "node_modules/**", "**/.obsidian/**"])
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_words: int = 350
    rrf_k: int = 60
    keyword_weight: float = 0.5
    vector_pool: int = 50
    section_pool: int = 400
    k: int = 20
    widen_at: float | str = "auto"
    widen_k: int = 150
    rare_max_df: int = 12
    max_windows: int = 24
    question_gate: bool = True
    question_kinds: bool = False
    widen_kinds: list[str] = field(default_factory=lambda: ["aggregate", "temporal"])
    reranker: dict = field(default_factory=lambda: {"backend": "cross-encoder"})
    max_usd_per_query: float = 0.05
    daily_cap_usd: float = 1.0
    source: Path | None = None

    @property
    def state_dir(self) -> Path:
        return self.index.parent

    @property
    def ledger(self) -> Path:
        return self.state_dir / "spend.jsonl"

    @property
    def recall_log(self) -> Path:
        return self.state_dir / "recall-log.jsonl"


def load(path: Path | None = None, **overrides) -> Config:
    path = path or find_config_file()
    raw: dict = {}
    if path and path.is_file():
        raw = tomllib.loads(path.read_text())
    base = path.parent if path else Path.home() / ".config" / "deeprecall"
    roots_env = os.environ.get("DEEPRECALL_ROOTS")
    roots = roots_env.split(os.pathsep) if roots_env else raw.get("roots", [])
    emb, srch, rec, bud = (raw.get(k, {}) for k in ("embedding", "search", "recall", "budget"))
    cfg = Config(
        roots=[_expand(r) for r in roots],
        index=_expand(os.environ.get("DEEPRECALL_INDEX") or raw.get("index") or base / "index.db"),
        exclude=raw.get("exclude", Config.__dataclass_fields__["exclude"].default_factory()),
        embed_model=emb.get("model", Config.embed_model),
        chunk_words=int(emb.get("chunk_words", Config.chunk_words)),
        rrf_k=int(srch.get("rrf_k", Config.rrf_k)),
        keyword_weight=float(srch.get("keyword_weight", Config.keyword_weight)),
        vector_pool=int(srch.get("vector_pool", Config.vector_pool)),
        section_pool=int(srch.get("section_pool", Config.section_pool)),
        k=int(rec.get("k", Config.k)),
        widen_at=rec.get("widen_at", "auto"),
        widen_k=int(rec.get("widen_k", Config.widen_k)),
        rare_max_df=int(rec.get("rare_max_df", Config.rare_max_df)),
        max_windows=int(rec.get("max_windows", Config.max_windows)),
        question_gate=bool(rec.get("question_gate", True)),
        question_kinds=bool(rec.get("question_kinds", False)),
        widen_kinds=_str_list(rec.get("widen_kinds", Config.__dataclass_fields__["widen_kinds"].default_factory())),
        reranker=dict(raw.get("reranker", {"backend": "cross-encoder"})),
        max_usd_per_query=float(bud.get("max_usd_per_query", Config.max_usd_per_query)),
        daily_cap_usd=float(bud.get("daily_cap_usd", Config.daily_cap_usd)),
        source=path,
    )
    if os.environ.get("DEEPRECALL_RERANKER"):
        cfg.reranker = {**cfg.reranker, "backend": os.environ["DEEPRECALL_RERANKER"]}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
