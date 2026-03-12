# ADR-002: Recall@k vs NDCG vs MRR — which metric matters when

## Status

Accepted.

## Context

The harness reports `recall@k`, `NDCG@k`, `MRR`, and `hit_rate@k` for every
run. Reviewers occasionally ask: which one is "the" metric? They are not
interchangeable — each measures a different thing and is appropriate for
different use cases. Picking one as canonical without context is misleading.

## Decision

Report all four. Treat them as answering different questions:

* **`recall@k`** — *Did the retrieval surface every positive within the
  top-k?* Defined as `|top_k ∩ positives| / |positives|`. The right metric
  when downstream processing (reranker, RAG context) gets to look at the
  whole top-k, because what matters is whether all relevant docs are in
  the candidate pool, not where they sit. Headline metric for the README
  charts.

* **`NDCG@k`** — *Are the positives in the right rank order?* Standard
  IR metric using `(2^rel - 1) / log2(rank + 1)` gain. The right metric
  for graded relevance (BEIR datasets that grade 0–3 instead of binary)
  and for systems where rank order is consumed directly (search results
  page, no reranker). The bench reports it whenever qrels carry graded
  relevance.

* **`MRR`** — *How quickly does the first positive appear?* `1 / rank`
  of the first positive. The right metric when a single hit is enough
  (a question-answering system that trusts the top-1, or a retrieval
  step before a powerful generator). MRR@10 is the BEIR convention.

* **`hit_rate@k`** — *Did the top-k contain at least one positive?*
  Binary 0/1. Useful as a sanity check. Goes to 1.0 fast on easy
  datasets; not informative once it's saturated.

## Default

Per-query metrics: all four are computed if the qrels support them.
Aggregates in the bench summary: `recall@k` (mean + p50) and `NDCG@k`
(mean) are surfaced because they are the two most cited in the BEIR /
MS-MARCO literature. `MRR` and `hit_rate` are available in the per-query
timing parquet if a reviewer wants them.

## Consequences

* Reports look noisier (four metrics not one). That is the point — a
  reader who sees "qdrant has 100% recall@10 at 8 ms p95" should also
  see "but NDCG@10 is the same as pgvector once you normalize"; otherwise
  you can claim a Qdrant win that isn't there.
* The graded-relevance path on BEIR (`relevance == 2`, `relevance == 3`)
  flows through unchanged because both `recall_at_k` and `ndcg_at_k`
  already accept any non-negative grade and exclude `relevance == 0`
  rows from the positive set.
