# ADR-003: HNSW vs IVFFLAT vs IVF-PQ — when each index makes sense

## Status

Accepted.

## Context

The bench supports three ANN index families across the four adapters.
The right index depends on corpus size, RAM budget, and the recall /
latency tradeoff. Reviewers reasonably ask: why does this benchmark
test all three?

## Decision

Test all three because they answer different operational questions:

| Index     | When it wins                                         | Knobs that matter                          | Backend support |
| --------- | ---------------------------------------------------- | ------------------------------------------ | --------------- |
| HNSW      | Recall-sensitive, latency-sensitive, RAM available   | `m`, `ef_construction`, `ef_search`        | pgvector, Qdrant (only) |
| IVFFLAT   | Tunable recall, no graph in RAM, larger-than-RAM ok  | `lists`, `probes`                          | pgvector |
| IVF-PQ    | Memory-constrained, recall ≥ 0.9 acceptable          | `num_partitions`, `num_sub_vectors`, `nprobes` | LanceDB |

### HNSW

Hierarchical Navigable Small World. Builds a multi-layer proximity graph;
search greedy-descends through layers. Best recall at low latency in this
benchmark, at the cost of a fully-resident graph. Default for both
pgvector and Qdrant. The `ef_search` knob is the recall/latency dial —
sweep it during the bench.

### IVFFLAT

Inverted-file partitioning with a flat (full-precision) per-cell scan.
Build is fast and the index size is essentially `n * dim * 4`. Search
visits `probes` cells, so recall scales monotonically with `probes`.
Better than HNSW when the corpus is bigger than RAM (the graph would
swap thrash) and when you need a recall knob that is cheap to retune
without rebuilding.

### IVF-PQ

Inverted-file with product quantization on the residuals. The vector
matrix is compressed by `dim / num_sub_vectors`× — typically 16× — so
a 100M × 384 corpus that would need 150 GB at flat precision fits in
~10 GB. Recall takes a hit (typically caps at 0.9–0.95), but the corpus
is now in RAM. The right index when memory is the binding constraint.

## Consequences

* The bench's params dict is the single source of truth for which index
  the adapter builds. Sweeping `ef_search` over HNSW is one
  `BenchSpec` per value of `ef_search`; sweeping `probes` over IVFFLAT
  is one per value of `probes`. The runner doesn't know about index
  families — they're all just `params`.
* Adapter contract test (`tests/test_adapters/test_contract.py`) does
  not parameterize over index family because the contract is "every
  adapter responds to the lifecycle the same way", not "every adapter
  hits the same recall floor". Integration tests verify membership in
  top-k under a small fixture corpus; real recall@k floors are
  evaluated only via the bench harness on real data.
* IVFFLAT is pgvector-only and IVF-PQ is LanceDB-only because that's
  what those backends ship; trying to force a single index family
  across all adapters would erase the comparison this ADR exists to
  preserve.
