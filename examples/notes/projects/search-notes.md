---
title: Search engine notes
keywords: [bm25, embeddings, rrf]
---
# Search engine notes

## Background
Hybrid search combines keyword and vector retrieval. We fuse them with reciprocal rank fusion.

## Tuning
We tried several fusion constants. The keyword leg is weighted at half the vector leg.
In the end the fusion constant k was set to 60 after a sweep over 1, 10, 30, 60 and 100.

## The ingest-sync-daemon incident
On a Tuesday the ingest-sync-daemon stopped copying files for nine days before anyone noticed,
because its failure only showed up as a smaller-than-usual log. The fix was an alert on stale output.
