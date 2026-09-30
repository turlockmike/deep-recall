# Benchmarks

Every number Deep Recall publishes, what it measures, and how it compares. Everything on LongMemEval is
reproducible with [`bench/longmemeval/`](../bench/longmemeval/), which has step-by-step instructions and
the per-question results.

## Summary

| Corpus | Measure | Hybrid search | + reranker | Questions |
|---|---|---|---|---|
| LongMemEval-S | all evidence sessions in the top 5 | 87.2% | **97.4%** | 470 answerable |
| LongMemEval-S | any evidence session in the top 5 | 96.8% | **99.6%** | all 500 |
| LongMemEval-S | answered correctly, end to end (Claude Sonnet 5.5 reading the top 5) | — | 96% (48/50) | 50 |
| A real 5,200-note agent memory | right file ranked #1 | 61.6% | ~97% | 99 held-out |

"+ reranker" is `deeprecall recall`: the same hybrid search plus the rare-term leg's exact-match
candidates, scored by the `jev` reranker. LongMemEval runs use
[`contrib-prompt.txt`](../bench/longmemeval/contrib-prompt.txt). The 5,200-note run uses the default prompt.

## LongMemEval-S retrieval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024) gives each question its own
history of about 48 chat sessions and labels the sessions that hold the answer. Deep Recall indexes each
history as one Markdown file per session and is scored on which files it returns.

| | Hybrid search | + reranker |
|---|---|---|
| All evidence in the top 5 (470 answerable) | 410 (87.2%) | **458 (97.4%)** |
| All evidence in the top 10 (470 answerable) | 445 (94.7%) | 463 (98.5%) |
| Any evidence in the top 5 (all 500) | 484 (96.8%) | **498 (99.6%)** |
| An evidence session ranked #1 (470 answerable) | 410 (87.2%) | 458 (97.4%) |

Cost with `jev`: $2.64 for all 500 questions, $0.0053 per question. The per-type breakdown and every miss
are in the [benchmark guide](../bench/longmemeval/).

## Compared with other memory systems

Only retrieval results measure the same thing as the numbers above. These are the LongMemEval-S retrieval
figures other projects publish, each checked against the project's own repository:

| System | Retrieval | Any evidence in the top 5 (500 questions) | Misses | Source |
|---|---|---|---|---|
| agentmemory | BM25 + vectors (all-MiniLM-L6-v2) | 95.2% | 24 | [benchmark/LONGMEMEVAL.md](https://github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md) |
| MemPalace, raw | vector search | 96.6% | 17 | [README](https://github.com/MemPalace/mempalace), metric in `benchmarks/longmemeval_bench.py` |
| Deep Recall, search only | BM25 + vectors (bge-small-en-v1.5) | 96.8% | 16 | [results](../bench/longmemeval/results/s-500-contrib-prompt/) |
| **Deep Recall + reranker** | the same, plus the answerability reranker | **99.6%** | **2** | [results](../bench/longmemeval/results/s-500-contrib-prompt/) |

MemPalace also publishes 98.4% (hybrid mode, tuned on 50 questions and scored on the other 450) and at
least 99% (with an LLM reranker). Its own README calls the 98.4% the generalisable figure. Both use the
same `recall_any@5` measure. The agentmemory and MemPalace numbers are as each project reports them.
Deep Recall has not re-run their code.

Not in the table: results at a k other than 5, and results that do not state how recall is defined.

### End-to-end scores

Most published LongMemEval scores (Mem0, Hindsight, Zep, Supermemory, Mastra, Emergence and others) are
end-to-end answer accuracy. A memory system retrieves, a model the vendor chose answers, and another
model grades the answer. Those scores rise and fall with the answering model. Mastra, for example,
reports the same memory system at 84.2% answering with GPT-4o and 94.9% with gpt-5-mini, graded by GPT-4o
([source](https://mastra.ai/research/observational-memory)). They are not comparable with the
retrieval numbers above, or with each other when the models differ.

Deep Recall stops at the context. As a check that the retrieved context is enough to answer from, 50
questions (seed 0) were run end to end:

- Claude Sonnet 5.5 answered from the top 5 sessions, using LongMemEval's reader prompt.
- Claude Opus 5.5 graded each answer, using LongMemEval's per-type grading prompts.
- **48 of 50 were correct.** Both misses were counting questions where one evidence session sat at rank
  6 or 7. That run used the default prompt; the contribution prompt puts all evidence in the top 5 for
  both.
- It cost $2.88, which is $0.058 per question.

## A real agent memory

The reranker was built on a 5,200-file agent memory: working notes, meeting records and project docs.
The test was 99 held-out questions whose answer is one sentence past word 500 of a long file, graded by a
blind LLM judge.

| | Right file ranked #1 |
|---|---|
| Hybrid search | 61.6% |
| + rare-term leg + section-level reranking (`jev`, k=20) | ~97% |
| The same, k=50 | 99.0% |

This corpus is private, so these numbers cannot be reproduced. They show the reranker holding up on the
kind of data it is meant for: many near-duplicate notes on the same topics, where search alone falls to
61.6%.

A smaller smoke test on a 115-file, 200k-word slice of the same memory (12 questions):

| | #1 right | Time per question | Cost per question |
|---|---|---|---|
| Hybrid search | 11/12 | ~0.2 s | free |
| `cross-encoder` (local) | 11/12 | ~20 s on a busy 4-core box | free |
| `jev` | 10/12 exact-path (the 2 misses rank another note that states the same fact) | ~3 s | ~0.8¢ |

On a small corpus, search alone is already strong. The reranker pays off as the corpus grows and fills
with near-duplicates. The local cross-encoder is CPU-bound: set `threads` under `[reranker]` to your core
count.

## What these numbers do not show

- **Answer accuracy at scale.** End-to-end has been measured on 50 questions only.
- **The free local reranker on LongMemEval.** Every LongMemEval number here uses `jev`.
- **LongMemEval-M**, the harder variant with about 500 sessions per question.
- **Exact-path undercounting.** A file that states the answer but is not in the dataset's evidence list
  counts as a miss.
