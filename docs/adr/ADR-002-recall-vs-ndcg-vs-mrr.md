# ADR-002: Recall@k vs NDCG vs MRR — which metric matters when

## Status

Accepted.

## Context

The harness reports `recall@k` and `NDCG@k` for every run. Reviewers
occasionally ask: which one is "the" metric, and why aren't `MRR` /
`hit_rate@k` in there too? They are not interchangeable — each
measures a different thing, but for the corpus shapes this benchmark
actually targets, the two we report are sufficient.

## Decision

Report `recall@k` and `NDCG@k`. Intentionally omit `MRR` and `hit_rate@k`
because they collapse signal that the two reported metrics already
preserve.

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

### Why not MRR or hit-rate

* **`MRR`** is dominated by the rank of the first positive. For
  multi-positive qrels (the BEIR norm) it discards everything past the
  first hit, which `NDCG@k` already weights with a more graceful
  log-decay. Single-positive datasets reduce `MRR` to the inverse rank
  of the lone positive, which `NDCG@k` also captures (and which
  `recall@k` reduces to a binary on). Adding `MRR` would add a third
  number that moves with the other two.
* **`hit_rate@k`** saturates fast — on most datasets at the values of
  `k` we sweep, it sits at 1.0 for every adapter, which makes it useless
  as a discriminator. When it isn't saturated, it's a quantized version
  of `recall@k` (collapsing partial recall to 0/1).

## Consequences

* Reports look noisier than a one-metric headline (two numbers not one).
  That is the point — a reader who sees "qdrant has 100% recall@10 at
  8 ms p95" should also see "but NDCG@10 is the same as pgvector once
  you normalize"; otherwise you can claim a Qdrant win that isn't there.
* The graded-relevance path on BEIR (`relevance == 2`, `relevance == 3`)
  flows through unchanged because both `recall_at_k` and `ndcg_at_k`
  already accept any non-negative grade and exclude `relevance == 0`
  rows from the positive set.
* If a future workload genuinely needs `MRR` — e.g. a top-1-only QA
  pipeline where rank-of-first-hit is the optimization target — adding
  it is a small change in `metrics/retrieval.py` plus a column in the
  bench summary parquet. Not blocked by this ADR; just not included by
  default.
