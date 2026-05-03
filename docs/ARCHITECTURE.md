# Architecture

`vector-db-bench` is a four-stage pipeline. Each stage produces an artifact
on disk so any later stage can run from a re-checkout without re-doing the
earlier work, and so reviewers can verify the methodology by inspecting the
parquet outputs rather than re-running the whole pipeline.

```
       ┌────────┐    ┌──────────┐    ┌────────┐    ┌────────┐
   →   │ corpus │ →  │  embed   │ →  │  bench │ →  │  plot  │   →  charts
       └────────┘    └──────────┘    └────────┘    └────────┘
       parquet +      parquet +      parquet +     png + svg
       manifest       manifest       summary

```

## Stage boundaries

Every boundary is a directory of parquet files plus a `manifest.json` that
records the upstream fingerprint. Fingerprints are content-addressed
(`blake2b` over the canonical serialization of the data + provenance
metadata), and downstream stages refuse to load if the upstream fingerprint
on disk no longer matches the data on disk. That is the only mechanism that
detects "someone replaced corpus/ under saved vectors" without re-doing the
work.

### corpus

* `src/vdbbench/corpus/bundle.py` — the canonical container.
  Validates that qrel ids point to real passages/queries, rejects duplicate
  `(qid, pid)` pairs, coerces id dtypes, and freezes a fingerprint at
  construction so equality and hashing stay stable under in-place mutation
  of the underlying frames.
* `src/vdbbench/corpus/synthetic.py` — generates a deterministic bundle
  from random unit vectors with brute-force top-k qrels. Used by every
  test and by the smoke run because it has zero IO and known ground truth.
* `src/vdbbench/corpus/beir.py` — streams BEIR datasets (incl. MS-MARCO)
  with always-keep judged-pid sampling and deterministic reservoir
  sampling on the unjudged. Memory bound is `O(sample_size)` regardless
  of the source corpus size.

### embed

* `src/vdbbench/embed/encoder.py` — encoder protocol, deterministic
  `FakeEncoder` for tests, `EncodedBundle` (corpus + dense vectors), and
  parquet IO via `pa.FixedSizeListArray.from_arrays` (zero Python list
  materialization). The on-disk manifest snapshots both the bundle
  fingerprint and the bundle fingerprint *at encode time* so saving a
  bundle that was mutated after encoding fails loudly instead of silently
  pairing stale vectors with the new corpus.
* `src/vdbbench/embed/sentence_transformer.py` — gated optional encoder
  using sentence-transformers. Lazy-imported so the package stays
  importable on platforms without torch wheels.

### metrics

* `src/vdbbench/metrics/retrieval.py` — `recall@k`, `NDCG@k`, `MRR`,
  `hit_rate`, plus `aggregate(values)` for mean/p50/p95/std/n. Pure
  functions so the bench harness can call them on any list of timings or
  hits. Judged-negative qrels (`relevance == 0`) are explicitly excluded
  from positives.

### adapters

* `src/vdbbench/adapters/base.py` — `VectorStoreAdapter` protocol.
  Lifecycle is `setup → ingest → build_index → search* → teardown`,
  identical across every backend. Per-backend tuning lives in the
  `params: dict` passed to `setup`.
* `src/vdbbench/adapters/pgvector.py` — Postgres 17 + pgvector via
  psycopg3. HNSW + IVFFLAT supported, with a per-session `ef_search` /
  `probes` knob applied via `SET LOCAL`. ANALYZE-after-index is opt-out.
* `src/vdbbench/adapters/qdrant.py` — Qdrant via the standard client.
  HNSW knobs through `HnswConfigDiff`. Optional payload indexes for
  hybrid filter + ANN.
* `src/vdbbench/adapters/lancedb.py` — embedded LanceDB, IVF-PQ index.
* `src/vdbbench/adapters/chroma.py` — embedded Chroma, no-tuning baseline.

### bench

* `src/vdbbench/bench/runner.py` — drives every adapter through the same
  lifecycle. Produces two parquet tables:
  * `timings.parquet`: one row per `(db, params_hash, repeat, qid)` with
    measured latency and the retrieved pids. Reproducible from this row
    alone — recall and NDCG are recomputed downstream.
  * `summary.parquet`: one row per `(db, params_hash)` with aggregate
    stats (`mean`, `p50`, `p95`, `p99` latency; `recall@k`; `NDCG@k`;
    ingest throughput; index build time; index disk footprint).

  `BenchSpec.profile` is an opinionated shorthand:
  * `cold` — first-query latency (no warmup, single repeat)
  * `warm` — default; cache pre-warmed before timing
  * `p99` — tail-focused (50 warmup, 5 repeats)

### plot

* `src/vdbbench/plot/charts.py` — every chart reads from a single
  `summary.parquet`, so `make plots` is a cheap re-run after any
  bench change. Charts produced: pareto frontier, recall bars, latency
  bars, ingest bars, index disk bars, speedup-vs-chroma bars.

## Determinism

Three load-bearing invariants make every run reproducible:

1. **Corpus.** `CorpusBundle.fingerprint()` is order-invariant on rows
   and metadata-aware. Two constructions of the same data hash to the
   same value regardless of insertion order or shuffle.
2. **Embedding.** `FakeEncoder` is hash-based per text; the
   sentence-transformers encoder is deterministic given the same model
   weights. The `EncodedBundle` manifest pins the corpus fingerprint at
   encoding time so any later mutation of `corpus/` is caught at load.
3. **Bench.** Per-query timings and retrieved pids land in
   `timings.parquet`; recall/NDCG are recomputed from those tables, so
   reviewers can re-derive every aggregate stat without re-running the
   bench.

## Testing pyramid

* **Unit tests** (`tests/test_corpus`, `tests/test_embed`,
  `tests/test_metrics`, `tests/test_plot`) — pure-Python, no Docker.
  Run on every CI matrix entry.
* **Adapter contract tests** (`tests/test_adapters/test_contract.py`) —
  exercise the lifecycle through a reference brute-force adapter so
  protocol regressions can be caught on any platform.
* **Adapter integration tests** (`tests/test_adapters/test_*_integration.py`)
  — behind `pytest.mark.integration`, need real DB containers. Skipped
  by default on developer machines.
* **CLI smoke** (`tests/test_smoke.py`) — `python -m vdbbench.cli version`
  works, version metadata lands, `vdbbench bench --help` includes the
  `--all` flag (so the documented `make bench` reproduction is wired
  through end-to-end).

## Demo vs full results

`results/demo/` ships in the repo with parquet + `bench_manifest.json`
from a 5 000-vector synthetic run that finishes in ~30 seconds. It's
the dataset the README headline numbers and `BLOG.md` story are
captured against. It is **not** the canonical benchmark.

The full sweep is a 1 M-vector MS-MARCO run produced by `make bench-1m`
into `results/full/` — too large to commit, and the numbers vary by
host. The recipe (`prep --dataset msmarco --sample-size 1000000`,
`bench --all --profile p99`, `plot`) is in the README so the run is
exactly reproducible. Every published parquet carries a
`bench_manifest.json` (schema_version=1) with encoder identity,
adapter versions, and host metadata so a reviewer can verify what
produced the numbers.
