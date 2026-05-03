# vector-db-bench

> Reproducible side-by-side benchmarks of **pgvector**, **Qdrant**, **LanceDB**, and **Chroma** on the same corpus, hardware, and queries — published with raw data and a one-command reproduction.

[![ci](https://github.com/BishBish123/vector-db-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/vector-db-bench/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![reproducible](https://img.shields.io/badge/reproducible-make%20bench-brightgreen)](Makefile)

---

## Why

Most vector-DB comparisons online are vendor blog posts or synthetic micro-benchmarks. This repo is built so a reviewer can:

1. Read the methodology and find no holes
2. Run `make bench-all` on a laptop and reproduce the **demo** numbers (and `make bench-1m` for the full sweep)
3. Re-run the analysis themselves from the published parquet

That's the bar.

> **Demo vs full.** The numbers in this README and the parquet checked into `results/demo/` come from a 5 000-vector synthetic sanity sweep — small enough to ship in the repo and re-run on a laptop in seconds. The canonical full sweep is a 1 M-vector MS-MARCO run produced by `make bench-1m` and persisted to `results/full/summary.parquet` (not committed; the recipe is below). When the README cites "Pareto frontier" or "p95 latency", it means the demo unless explicitly tagged `[full]`.

## Headline chart (5K-vector demo)

The numbers below come from a 5 000-vector synthetic corpus with brute-force ground truth, run on a 2020 Intel MacBook Air against pgvector pg17 + qdrant 1.17 in Docker. They're a sanity-check of the pipeline, not the canonical benchmark — `make bench-1m` is what produces the full 1 M-vector MS-MARCO sweep.

![Pareto frontier — recall vs p95 latency](assets/pareto.png)

| DB | Ingest (vps) | p95 latency (ms) | Recall@10 | NDCG@10 | QPS (est.) | Index disk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector (HNSW defaults) | 7 450 | 57 | 0.864 | 0.909 | ~21 | 4.5 MB |
| qdrant (HNSW defaults) | 5 534 | 8 | 1.000 | 1.000 | ~143 | (mem) |

Notes worth flagging — these are *exactly* the kinds of caveats the blog post will dig into:

- `pgvector` numbers above include per-query connection-open overhead (~40 ms on macOS Docker). A connection-pooled adapter would close most of the latency gap; both numbers are honest as measured today.
- 100 % recall on Qdrant at this scale is expected — HNSW with default `m=16` over 5 000 vectors is essentially exact. The interesting curves come from sweeping `ef_search` over a real corpus.

## Reproduce

### Demo (the numbers in this README)

```bash
git clone https://github.com/BishBish123/vector-db-bench.git
cd vector-db-bench

# 1. Install (uv-managed, all extras the platform supports).
make install

# 2. Bring up pgvector + qdrant in Docker.
make up

# 3. Demo run — 5 000 synthetic vectors, ~30 seconds end-to-end.
uv run vdbbench prep   --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
uv run vdbbench bench  --encoded data/encoded-demo --out results/demo \
                       --pgvector-dsn postgresql://bench:bench@localhost:5433/bench \
                       --qdrant-url http://localhost:6333
uv run vdbbench plot   --summary results/demo/summary.parquet --out assets
```

`results/demo/summary.parquet` is checked into the repo so a reviewer can run only step 4 (the plot) and inspect the published numbers without bringing services up. `bench_manifest.json` next to it captures encoder identity, adapter versions, and host metadata for the run that produced those numbers.

### Full (1 M MS-MARCO, the canonical benchmark)

The full sweep is reproducible but **not** checked in — the parquet would be too large and the recall numbers are tied to the host the run was performed on. The recipe:

```bash
uv run vdbbench prep  --dataset msmarco --sample-size 1000000 --out data/encoded-1m
uv run vdbbench bench --encoded data/encoded-1m --out results/full --all --profile p99
uv run vdbbench plot  --summary results/full/summary.parquet --out assets/full
```

`make bench-1m` (planned) is the one-command wrapper. Expect ~6–12 hours wall-clock on a laptop depending on which adapters land — the Pareto sweep over `ef_search` / `probes` / `nprobes` is what eats the time. Persist the resulting `results/full/` tree (parquet + `bench_manifest.json`) when you publish numbers.

The `--lancedb-path` and `--chroma-path` flags (used by `--all`) are skipped on Intel macOS (no wheels for `lancedb` / `chromadb`+`onnxruntime`); use the Docker bench image or run on Linux / arm64 macOS for the full four-way comparison.

## Methodology

- **Corpora.** MS-MARCO via [BeIR/msmarco](https://huggingface.co/datasets/BeIR/msmarco) (sample-size capped, all judged passages always kept), or any other BEIR dataset (`scifact`, `nfcorpus`, `fiqa`, …). A pure-Python synthetic corpus with brute-force ground truth is included for CI and smoke tests.
- **Embedding model.** `BAAI/bge-small-en-v1.5` (384-dim) by default; pluggable via `--embed-model`.
- **Hardware.** Single machine, documented per run. No GPU unless the run says so.
- **Fairness.** Per-spec warm-up queries (default 10) are discarded before timing; `repeats > 1` runs the full query set multiple passes.
- **Repeats.** Per-(db, params) summary aggregates across `repeats × n_queries` measured timings.
- **Recall denominator.** The `relevance > 0` qrels are the positives; judged-negative entries (`relevance == 0`) are explicitly excluded from recall and NDCG.

The full sweep of HNSW `ef_search`, IVF `lists`/`probes`, and IVF-PQ `num_partitions` lands in a follow-up commit (the harness plumbs `params` straight to the adapters; the missing piece is a `vdbbench sweep` runner that emits a list of `BenchSpec` values across a knob grid).

## Stack

| Layer | Choice |
| --- | --- |
| Embeddings | `sentence-transformers` (bge-small, nomic) |
| Corpus | BEIR via Hugging Face `datasets` (streaming + judged-aware sampling) |
| pgvector | Postgres 17 + `pgvector` 0.8.0 in Docker (image tag `pgvector/pgvector:0.8.0-pg17`) |
| Qdrant | Qdrant 1.17 in Docker |
| LanceDB | embedded (no service) |
| Chroma | embedded persistent client |
| Harness | Python 3.11+, `pytest-benchmark` |
| Plots | matplotlib + seaborn |
| Reproducibility | `docker compose` with pinned versions, `make` targets, parquet outputs |

## Platform support

| Platform | Local dev (`make install`) | Local bench |
| --- | --- | --- |
| Linux x86_64 | ✅ all extras | ✅ |
| macOS arm64 (Apple Silicon) | ✅ all extras | ✅ |
| macOS x86_64 (Intel) | ✅ core + dev only (`make install-min`) | ⚠️ pgvector + qdrant only — lancedb / chroma need the Docker bench image |

Windows is unsupported (`Makefile` uses bash). WSL2 works.

> **Why the Intel-Mac caveat?** `torch` (and therefore `sentence-transformers`), `chromadb` (via `onnxruntime`), and `lancedb` no longer ship macOS x86_64 wheels.

## What this benchmark does NOT measure

- Filtered search (`WHERE category = 'X' AND vector ≈ q`) — partially supported: Qdrant exposes filter+ANN through `QdrantAdapter.search(query, k, filter=...)` (see `tests/test_adapters/test_qdrant_integration.py::test_filter_passthrough_subsets_results`); pgvector / lancedb / chroma adapters reject a non-`None` filter explicitly. A cross-adapter sweep is the Phase 6 stretch.
- Hybrid search (BM25 + vector) — Phase 6 stretch.
- Multi-tenancy at scale.
- Geo-replicated reads.
- Index recovery time after a crash.
- GPU acceleration.

These are real questions; they're omitted on purpose to keep the comparison apples-to-apples.

## Layout

```
src/vdbbench/
  corpus/       BEIR + synthetic loaders, CorpusBundle (parquet IO + sampling)
  embed/        Encoder protocol, FakeEncoder, sentence-transformers wrapper
  metrics/      recall@k, NDCG, MRR, hit-rate, aggregator
  adapters/     pgvector / qdrant / lancedb / chroma
  bench/        run_bench(): drives every adapter through one lifecycle
  plot/         pareto + per-axis bar charts
  cli.py        `vdbbench prep | bench | plot`

tests/          200+ unit tests (all green) + integration tests behind
                pytest.mark.integration (skip on Intel macOS for lance/chroma)

docs/
  ARCHITECTURE.md        layered design + determinism invariants
  adr/
    ADR-001-...           why these four adapters
    ADR-002-...           recall@k vs NDCG vs MRR
    ADR-003-...           HNSW vs IVFFLAT vs IVF-PQ
    ADR-004-...           why ship a deterministic FakeEncoder
```

See [BLOG.md](BLOG.md) for the writeup of one specific tradeoff this bench surfaced, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layered design, and the [docs/adr/](docs/adr/) directory for the design decisions.

## License

MIT. See [LICENSE](LICENSE).
