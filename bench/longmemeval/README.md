# LongMemEval benchmark

Does deep-recall load the right context, the context from which a model *could* answer? This benchmark
measures that directly. No model answers and no judge grades, so the score is a plain count you can check
against the dataset's labels.

## Results

LongMemEval-S, all 470 answerable questions, `jev` reranker with [`contrib-prompt.txt`](contrib-prompt.txt):

| | Hybrid search | + jev rerank |
|---|---|---|
| **All evidence sessions in the top 5** | 410 / 470 (87.2%) | **458 / 470 (97.4%)** |
| All evidence sessions in the top 10 | 445 / 470 (94.7%) | 463 / 470 (98.5%) |
| An evidence session ranked #1 | 410 / 470 (87.2%) | 458 / 470 (97.4%) |
| Any evidence session in the top 5, all 500 questions | 484 / 500 (96.8%) | **498 / 500 (99.6%)** |

The last row includes the 30 abstention questions, which other published LongMemEval-S recall figures
also score. [docs/benchmarks.md](../../docs/benchmarks.md) compares these numbers with other memory systems.

| Question type | n | Search #1 | Rerank #1 | All evidence in top 5 |
|---|---|---|---|---|
| single-session-user | 64 | 53 | 64 | 64 |
| single-session-assistant | 56 | 56 | 56 | 56 |
| knowledge-update | 72 | 68 | 71 | 72 |
| single-session-preference | 30 | 21 | 25 | 29 |
| temporal-reasoning | 127 | 104 | 122 | 122 |
| multi-session | 121 | 108 | 120 | 115 |

Cost: **$2.43** for the run ($0.0052 per question, 57.8M jev input tokens at $0.042 per million). The reranker
widened to the whole haystack on 119 of 470 questions. Measured 2026-09-29 at deep-recall commit `bbbeeee`,
with an earlier version of `run.py` that recorded fewer fields per question (see [Committed results](#committed-results)).

### Which reranker prompt

The default `jev` prompt asks whether a passage *states the answer*. A question such as "how many hours did I
drive in total?" has no single passage that does, so every partial fact scored at the 0.01 floor, level with
unrelated chats. `contrib-prompt.txt` asks whether a passage holds *part of* the answer. On the same 50
questions (seed 0):

| Prompt | Evidence #1 | All evidence in top 5 | Widened | Cost |
|---|---|---|---|---|
| default (`rerankers/jev.py`) | 48 / 50 | 48 / 50 | 30 | $0.315 |
| `contrib-prompt.txt` | 48 / 50 | **50 / 50** | 12 | $0.260 |

Both prompts ranked the same file first on every single-fact question. The gains are on multi-session
counting questions, where the missing sessions moved from ranks 6 and 7 into the top 3. The default prompt
stays the default until the contribution prompt is measured on a corpus with near-duplicate notes, the case
the default's "not merely the same topic" clause exists for.

## What is measured

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813))
tests long-term memory in chat assistants. In the S variant, each of 500 questions comes with its own
history of about 48 chat sessions (roughly 115k tokens). The answer is usually one remark the user made in
passing, and the dataset labels which sessions contain it (`answer_session_ids`).

For each question the harness:

1. writes each session to its own Markdown file (`<session_id>.md`, with the session date in the body),
2. builds a fresh deep-recall index over those files,
3. runs `deeprecall search` (hybrid keyword + vector search) and `deeprecall recall` (the same search plus
   the rare-term leg's exact-match candidates, scored by the reranker), keeping the top 10 of each,
4. scores both lists against `answer_session_ids`.

| Metric | Meaning |
|---|---|
| `all@5` | every evidence session is in the top 5. The headline: a model given these five files has everything it needs. |
| `all@10` | every evidence session is in the top 10 |
| `any@5` | at least one evidence session is in the top 5 |
| `any@1` | the #1 file is an evidence session |

The 30 abstention questions (`question_id` ending in `_abs`) ask about something never said, so LongMemEval's
own retrieval evaluation excludes them. The dataset still labels related sessions for them, and other
projects' published recall figures score all 500. `--include-abstention` scores them too.

## Reproduce it

Requires Python 3.10+, [`uv`](https://docs.astral.sh/uv/), about 2.5 GB of free disk, and, for the `jev`
backend, a [TypeSafe](https://docs.typesafe.ai) API key.

**1. Install deep-recall**

```bash
git clone https://github.com/turlockmike/deep-recall && cd deep-recall
uv sync                            # installs from uv.lock
export DEEPRECALL_BIN="$PWD/.venv/bin/deeprecall"
```

**2. Download the dataset**

```bash
curl -L -o longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
shasum -a 256 longmemeval_s_cleaned.json
# d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
```

The file is 277 MB. The results above were measured against this checksum.

**3. Run one question first**

```bash
export TYPESAFE_API_KEY=...
python bench/longmemeval/run.py runs/smoke --data longmemeval_s_cleaned.json --n 470 \
  --prompt-file bench/longmemeval/contrib-prompt.txt --max-usd 0.001
```

`--max-usd` stops before the next question once spend reaches it, so this runs exactly one question
(about $0.005). The first run also downloads the embedding model. Check the stderr line reads
`"mode": "rerank:jev"`; any other mode means the reranker did not run and the question fell back to search order.

**4. Run all 470**

```bash
python bench/longmemeval/run.py runs/s-470 --data longmemeval_s_cleaned.json --n 470 \
  --prompt-file bench/longmemeval/contrib-prompt.txt --max-usd 4
```

About 4 hours on an Apple M4 Pro: ~21 s to index each haystack, ~8 s to rerank. Re-running the same
command resumes where it stopped. A run directory is tied to its settings (`run.json`), so a different
`--n`, `--seed`, `--backend` or prompt needs a new directory. Omit `--prompt-file` to use the default
prompt, and use a smaller `--n` for a stratified sample (seeded by `--seed`, default 0). For all 500
questions, add `--include-abstention` and set `--n 500`.

The run stops with exit 1 if any question's reranker did not run (for example, a missing or expired API
key), rather than scoring search order as reranked recall.

**5. Read the results**

`runs/s-470/summary.json` holds the totals and a per-type breakdown (also printed on stdout).
`rows.ndjson` holds one line per question:

| Field | Content |
|---|---|
| `qid`, `type`, `gold` | question id, question type, evidence session files |
| `search`, `recall` | top-10 session files from first-stage search and from reranked recall |
| `scores` | reranker score for each file in `recall` |
| `search_score`, `recall_score` | the four metrics for each list |
| `mode`, `widened`, `usd`, `tokens` | how recall ran, whether it widened, and what it cost |
| `pool`, `note`, `secs_recall`, `secs_index` | files reranked, recall's note, and timings |

To see the misses: `jq -c 'select(.recall_score["all@5"] | not) | {qid, type, gold, recall}' runs/s-470/rows.ndjson`

### Other rerankers

`--backend` takes any deep-recall backend. `cross-encoder` runs locally for free, and `openai` works with
any OpenAI-compatible endpoint that returns logprobs; configure either as in the main README. Neither has
been measured on this benchmark. `--prompt-file` applies to `jev` only and is refused for other backends.

Each deeprecall call runs with the other `DEEPRECALL_*` variables removed, so a `DEEPRECALL_CONFIG`
pointing at your own notes cannot leak into the benchmark.

## Reading the numbers

- **Exact-path scoring undercounts.** On "What time do I wake up on Saturday mornings?" the #1 file says "the
  previous Saturday, I woke up at 7:30 am", which is the answer, but it is not a labeled evidence session.
- **LongMemEval-S is easy for retrieval.** Most sessions in a haystack are about unrelated topics, and hybrid
  search alone puts an evidence session in the top 5 for 455 of 470 questions. The harder variant,
  LongMemEval-M, has about 500 sessions per question and is where the paper reports its retrieval numbers.
- **Evidence count limits `all@5`.** Three questions have six evidence sessions, so they cannot score on
  `all@5` at all. Of the 467 that can, 458 do.
- **Runs agree up to ties.** Re-running a question with this `run.py` at the current commit reproduced its
  top ranks, cost and metrics. Files tied at the reranker's 0.01 floor can swap places, and individual
  scores can move by about 0.01.
- **This is retrieval, not answering.** It shows the evidence reached the context, not that a model used it
  correctly. LongMemEval's published leaderboard scores end-to-end answers, so the two numbers are not
  comparable.
- On macOS, onnxruntime can abort (exit −6) while a finished index process shuts down. The harness retries
  the index once; `"event": "index_retry"` lines in stderr record each retry.

## Committed results

[`results/`](results/) holds the four runs quoted above, each as `summary.json` plus `rows.ndjson`
(ids, rankings and metrics; no dataset text).

| Directory | Run | Per-question fields the original run did not record |
|---|---|---|
| `s-470-contrib-prompt` | all 470, contribution prompt | `scores` (only `top_score`), `pool`, `note`, `secs_index` |
| `s-50-default-prompt` | 50 (seed 0), default prompt | `scores` (only `top_score`), `pool`, `note`, `secs_index` |
| `s-50-contrib-prompt` | the same 50, contribution prompt | `tokens`, `secs_recall`, `pool`, `note`, `secs_index` |
| `s-500-contrib-prompt` | all 500 (`--include-abstention`): the 470 above plus the 30 abstention questions | as `s-470-contrib-prompt` for the 470; none for the 30 |

A run of the current `run.py` records every field in the table in step 5.
