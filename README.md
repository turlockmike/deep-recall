# Deep Recall

**Answer-aware memory for Markdown notes, as a Claude Code plugin.**

Most note search finds files *about* your question. Deep Recall finds the file that **states the
answer**. A fast hybrid search (keyword + embeddings + an exact rare-term leg) gathers candidates.
Then a pluggable **reranker** reads each candidate section and scores one thing: *does this
passage state the answer?*

```
$ deeprecall recall "Which electricity plan did we switch to?"
1. 0.90  home/utilities.md
         § Home utilities > Electricity
2. 0.01  projects/journal-3.md

--- #1 winning section (home/utilities.md) ---
## Electricity
... After a long call we switched to the **Saver Plus 12** plan: 12.76 cents per kWh ...
```

## Why

On a real 5,200-file agent memory, measured on questions whose answer is one sentence buried past
word 500 of a long file (held-out set, 99 questions, blind LLM judge):

| | right file ranked #1 |
|---|---|
| hybrid search alone | 61.6% |
| + rare-term leg + section-level answerability reranking (jev backend, k=20) | ~97% |
| same, k=50 | 99.0% |

The write-up covers how each piece earns its place, what it costs, and how the evaluation was run.
Those numbers are from one corpus and one reranker (TypeSafe jev). The free local cross-encoder
backend is weaker on buried facts but still well ahead of search alone. Run `deeprecall eval` on
your own notes to see where you land.

## Install

### As a Claude Code plugin (private repo)

```bash
claude plugin marketplace add turlockmike/deep-recall
claude plugin install deep-recall@deep-recall
```

Then, in Claude Code: `/deep-recall:setup ~/notes`. This writes a config, builds the index and runs
a first lookup. Requirements: [`uv`](https://docs.astral.sh/uv/), plus `ripgrep` (optional, makes
the rare-term leg faster). The plugin's `bin/deeprecall` creates its own Python environment on
first use.

The plugin provides:

| Piece | What it does |
|---|---|
| MCP server `deeprecall` | tools `recall(question)`, `search(query)`, `reindex()` Claude calls natively |
| skill `recall` | tells Claude when to look things up in your notes and how to read the scores |
| `/deep-recall:recall <question>` | one-shot lookup |
| `/deep-recall:setup [folder]` | guided config + first index |
| `/deep-recall:reindex` | refresh after edits |
| SessionStart hook | incremental re-index in the background (only when a config exists) |
| `deeprecall` on PATH | the CLI, while the plugin is enabled |

### As a Python package / CLI

```bash
uv tool install 'git+ssh://git@github.com/turlockmike/deep-recall[mcp]'   # or: pip install -e '.[mcp]'
deeprecall init --root ~/notes
deeprecall index
deeprecall recall "When is the lawn service scheduled?"
```

## Commands

```
deeprecall init [--root DIR ...] [--backend NAME] [--here]   write config (global, or ./.deeprecall/)
deeprecall index [--full]                                     build / incrementally update
deeprecall search "terms" [-k 10]                            first-stage hybrid search (free, fast)
deeprecall recall "question?" [--top 5] [--backend NAME]     answer-aware recall
deeprecall status                                             config, index size, reranker, spend
deeprecall rerankers                                          list reranker backends
deeprecall eval questions.jsonl [--limit N] [--max-usd 1]    hit@1 / hit@3: search vs recall
```

Only question-shaped queries (ending in `?`, or starting with a question word) are reranked. Keyword
lookups ("electricity plan") return search order, free and instant, because "does this passage
state the answer?" is ill-posed for them.

## How it works

1. **Index (SQLite).**
   - Keyword: FTS5 (`porter unicode61`).
   - Vectors: sqlite-vec with [fastembed](https://github.com/qdrant/fastembed) `BAAI/bge-small-en-v1.5` (384-d, ONNX, CPU).
   - File vector: the mean of its 350-word chunk vectors. Frontmatter `summary`/`description` is added as an extra chunk.
   - Files longer than 350 words are also indexed **section by section**, split at Markdown headings.
   - Incremental: re-indexes only files whose content hash changed.
2. **First stage.**
   - Keyword leg: BM25.
   - Vector leg: a long file is scored by its *best section*, not a diluted file average.
   - Fusion: reciprocal rank fusion (k=60, keyword weight 0.5).
3. **Rare-term leg.**
   - Pulls distinctive tokens out of the question: `tool-names`, `IDs_like_this`, ALLCAPS, CamelCase, letter+digit codes, capitalised phrases, quoted strings.
   - Exact-matches them with ripgrep, keeping terms found in ≤ 12 files.
   - Pairs of common terms that are rare *together* also count.
   - Embeddings blur identifiers and BM25 splits them; this leg fixes most first-stage misses.
4. **Rerank.**
   - Each candidate file is cut into heading-bounded windows, sized to the reranker (≤ 24 per file).
   - The reranker scores every window as P(states the answer), and a file takes its best window.
5. **Widen when unsure.** If the best score is below the backend's threshold, widen once to the top 150 and score only the new files.
6. **Fail open.** Any reranker error (network, auth, budget) returns first-stage order with a note.

## Rerankers (plug in any "jev-like" model)

| backend | where it runs | cost | notes |
|---|---|---|---|
| `cross-encoder` (default) | local CPU (fastembed ONNX) | free | `Xenova/ms-marco-MiniLM-L-6-v2`; also `-L-12-v2`, `BAAI/bge-reranker-base`, `jinaai/jina-reranker-v2-base-multilingual` |
| `jev` | [TypeSafe](https://docs.typesafe.ai) API | ~0.3–1.5¢/question | calibrated yes/no decision model; strongest tested |
| `openai` | any OpenAI-compatible endpoint with logprobs | provider price | OpenAI, OpenRouter, Together, Groq, vLLM, llama.cpp, LM Studio… P(yes) from first-token logprobs |
| `command` | any executable | yours | JSON on stdin → scores on stdout; any language |
| `none` | — | free | first-stage order only |

```toml
[reranker]
backend = "jev"                  # needs TYPESAFE_API_KEY
# widen_at = 0.7                 # override the backend's confidence threshold
# window_words = 600
```

```toml
[reranker]
backend = "openai"
base_url = "http://localhost:8000/v1"   # e.g. a local vLLM
model = "Qwen/Qwen2.5-7B-Instruct"
api_key_env = "OPENAI_API_KEY"
usd_per_mtok = 0.0
```

```toml
[reranker]
backend = "command"
command = ["python3", "/path/to/my_scorer.py"]   # reads {"question","passages":[{path,heading,text}]}
                                                   # prints [0.93, 0.02, ...]
```

### Writing a reranker in Python

```python
from deeprecall.rerankers.base import Reranker, Passage

class MyReranker(Reranker):
    name = "mine"
    window_words = 400      # passage size you want
    widen_at = 0.6          # below this, deeprecall widens the pool once
    usd_per_token = 0.0     # for the spend meter

    def __init__(self, **options):          # the [reranker] table
        ...

    def score(self, question: str, passages: list[Passage]) -> list[float]:
        # return P(passage states the answer) in [0, 1], one per passage
        ...
```

Register it with an entry point, `[project.entry-points."deeprecall.rerankers"] mine = "pkg.mod:MyReranker"`,
or set `backend = "pkg.mod:MyReranker"` directly.

What mattered in testing:
- **Score each passage independently.** Asking a model to pick the best of N files was worse than scoring files one by one.
- **Ask about answerability, not relevance:** "does this passage *explicitly state* the fact that answers the question?"

## Cost control

Every paid call is estimated before it's sent and recorded in `spend.jsonl` next to the index.
Two caps apply, and a refusal falls back to first-stage order:

```toml
[budget]
max_usd_per_query = 0.05
daily_cap_usd = 1.00        # rolling 24h across all processes; 0 disables
```

`deeprecall status` shows spend for the last 24 hours and all time.

## Evaluate on your own notes

```jsonl
{"q": "Which electricity plan did we switch to?", "gold": ["home/utilities.md"]}
```

```bash
deeprecall eval my-questions.jsonl --limit 30 --max-usd 0.50
```

Exact-path scoring undercounts when another note states the same fact in other words. For headline
numbers, have a blind judge check the non-gold top hits.

## Config reference

`deeprecall init` writes a commented template. It lives at `./.deeprecall/config.toml` (with
`--here`) or `~/.config/deeprecall/config.toml`. The first one found walking up from the current
directory wins. `$DEEPRECALL_CONFIG` overrides the lookup.
Environment overrides: `DEEPRECALL_ROOTS` (path-separated), `DEEPRECALL_INDEX`, `DEEPRECALL_RERANKER`.

## Limits

- Markdown only (`*.md`).
- First index of a large corpus is CPU-bound: roughly 1–2k section embeddings per minute on a laptop.
- Hosted rerankers send note text to that provider. Use `cross-encoder` or a local `openai`-compatible server for private notes.

## Development

```bash
uv venv && uv pip install -e '.[mcp,dev]'
.venv/bin/pytest -q                      # offline: hash embedder + fake rerankers
claude plugin validate . --strict
claude --plugin-dir .                    # load the plugin for one session
```
