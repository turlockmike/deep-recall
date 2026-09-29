"""MCP server exposing deeprecall to Claude Code (or any MCP client) as native tools.

Tools:
  recall(question, top=5)   answer-aware recall; returns ranked files + the #1 winning section
  search(query, k=10)       first-stage hybrid search (free, fast)
  reindex()                 incremental index update
Requires the `mcp` extra:  pip install 'deeprecall[mcp]'
"""
from __future__ import annotations

import json


def main() -> None:
    try:                                    # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                     # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    from . import index as idx
    from .config import load
    from .recall import Recaller
    from .search import search as _search

    app = Server("deeprecall")
    state: dict = {}

    def recaller() -> Recaller:
        if "rc" not in state:
            state["rc"] = Recaller(load())
        return state["rc"]

    @app.tool()
    def recall(question: str, top: int = 5) -> str:
        """Find the note that STATES the answer to a question in the user's Markdown knowledge base.
        Ask a full, specific question (e.g. "Which rate plan did we switch the electricity to?").
        Returns ranked files with a 0-1 answerability score, the matching section heading, and the
        text of the best-matching section. Scores below ~0.5-0.7 mean no note clearly answers it."""
        r = recaller().recall(question, top=top)
        return json.dumps({"mode": r.mode, "widened": r.widened, "usd": r.usd, "note": r.note,
                           "results": [{"path": h.path, "score": h.score, "section": h.section} for h in r.hits],
                           "best_section_text": (r.hits[0].passage[:4000] if r.hits else "")}, indent=1)

    @app.tool()
    def search(query: str, k: int = 10) -> str:
        """Fast keyword + semantic search over the knowledge base (no reranking). Good for topic
        lookups ("printer toner") and listing related notes."""
        return json.dumps([{"path": p, "score": round(s, 5)} for p, s in _search(load(), query, k)], indent=1)

    @app.tool()
    def reindex() -> str:
        """Update the index after notes were added or edited (incremental; only changed files)."""
        state.pop("rc", None)
        return json.dumps(idx.build(load(), quiet=True))

    app.run()


if __name__ == "__main__":
    main()
