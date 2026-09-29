---
name: recall
description: Look things up in the user's Markdown notes / knowledge base / second brain (Obsidian vault, docs folder, journal, meeting notes) with deeprecall. Use whenever the user asks about something they wrote down, decided, logged or noted before ("what did we decide about…", "when did I…", "which plan did we pick", "find my note on…"), and before answering from memory about the user's own projects, preferences or history.
allowed-tools: Bash(deeprecall:*), mcp__plugin_deep-recall_deeprecall__recall, mcp__plugin_deep-recall_deeprecall__search, mcp__plugin_deep-recall_deeprecall__reindex
---

# Recall from the user's notes

deeprecall finds the note that **states the answer**, not just notes on the same topic. It runs a
hybrid keyword + semantic search, then a reranker reads each candidate section and scores whether
it states the answer.

## How to look something up

Prefer the MCP tool `recall` (plugin server `deeprecall`). If MCP isn't available, use the CLI:

```bash
deeprecall recall "Which electricity plan did we switch to?"
```

- **Ask a full, specific question**, ending with "?". Question-shaped queries are reranked;
  keyword queries ("electricity plan") return plain search order, which is fine for browsing a topic.
- Output: ranked files with a 0–1 score and the best-matching section heading, then the text of the
  #1 file's winning section. Quote from that text; open the file only if you need more context.
- **Score ≥ ~0.7** (jev / LLM backends) or **≥ ~0.5** (local cross-encoder): the note states the answer.
  Below that, say you couldn't find it clearly. Don't guess from a low-scoring note.
- Topic browsing / listing related notes: `deeprecall search "terms" -k 10` (free, instant).

## First run

If `deeprecall` reports no config or no index:

1. Ask the user which folder(s) hold their notes (or use one they already named).
2. `deeprecall init --root <folder>` (add `--here` to keep config + index in `./.deeprecall/`).
3. `deeprecall index` (first run downloads a ~70 MB embedding model; later runs only touch changed files).

The plugin refreshes the index in the background at session start. After editing notes mid-session,
run `deeprecall index` (or the `reindex` tool).

## Cost

The default reranker is a local cross-encoder: free and private. Hosted backends (jev, any
OpenAI-compatible LLM) are more accurate on buried facts and cost roughly 0.3–1.5¢ per question;
every call is metered with a per-query and daily cap (`deeprecall status` shows spend).
