# vector-db-bench

> Reproducible side-by-side benchmarks of **pgvector**, **Qdrant**, **LanceDB**, and **Chroma** on the same corpus, hardware, and queries — published with raw data and a one-command reproduction.

[![ci](https://github.com/BishBish123/vector-db-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/vector-db-bench/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![reproducible](https://img.shields.io/badge/reproducible-make%20bench--all-brightgreen)](Makefile)

---

## Why

Most vector-DB comparisons online are vendor blog posts or synthetic micro-benchmarks. This repo is built so a reviewer can:

1. Read the methodology and find no holes
2. Run `make bench-all` on a laptop and reproduce the numbers
3. Download `results/raw.parquet` and re-do the analysis themselves

That's the bar.

## Status

🚧 In active development. Phase 1 (corpus + ground-truth) lands in the next commit.

| Phase | Status |
| --- | --- |
| 0 — scaffold (this commit) | ✅ |
| 1 — corpus + embeddings + qrels | ⏳ |
| 2 — adapter interfaces (pgvector, Qdrant, LanceDB, Chroma) | ⏳ |
| 3 — bench runner + sweeps | ⏳ |
| 4 — analysis + Pareto plots | ⏳ |
| 5 — blog post + final README | ⏳ |
| 6 — hybrid + filtered search (stretch) | ⏳ |

## Reproduce

Today (Phase 0 — scaffold only):

```bash
git clone https://github.com/BishBish123/vector-db-bench.git
cd vector-db-bench
make install        # uv sync (all extras where wheels exist)
make check test     # lint + typecheck + smoke tests
```

Once Phase 3 lands:

```bash
make up             # docker compose up -d (pgvector + qdrant)  -- Phase 2
make bench-all      # ~6–10h on a laptop; emits results/raw.parquet + plots
```

## Methodology (will be expanded in Phase 5)

- **Corpus.** MS-MARCO passages (1M sample, deterministic seed) + Wikipedia 1M articles. Two embedding models: `BAAI/bge-small-en-v1.5` (384-dim) and `nomic-embed-text-v1.5` (768-dim).
- **Hardware.** Single machine, documented in `results/hardware.json` per run. No GPU unless the run says so.
- **Fairness.** 100-query warm-up per DB; first/last results dropped; same `k=10`; same query set; per-DB tuning budget capped and documented.
- **Repeats.** 5 runs per (db, params); high/low dropped; mean ± std reported.
- **Memory.** RSS sampled every 250 ms during query phase; report steady-state median + p95.

## Platform support

| Platform | Local dev (`make install`) | Local bench (Phase 2+) |
| --- | --- | --- |
| Linux x86_64 | ✅ all extras | ✅ |
| macOS arm64 (Apple Silicon) | ✅ all extras | ✅ |
| macOS x86_64 (Intel) | ✅ core + dev only (`make install-min`) | ✅ via Docker once Phase 2 lands |

Windows is unsupported (`Makefile` uses bash). WSL2 works.

> **Why the Intel-Mac caveat?** `torch` (and therefore `sentence-transformers`), `chromadb` (via `onnxruntime`), and `lancedb` no longer ship macOS x86_64 wheels. On Intel Macs, install only the core + dev tooling locally and run the bench inside the Docker image that lands in Phase 2.

## Stack

| Layer | Choice |
| --- | --- |
| Embeddings | `sentence-transformers` (bge-small, nomic) |
| Corpus | MS-MARCO via Hugging Face `datasets`, BEIR qrels |
| pgvector | Postgres 17 + `pgvector` 0.8 in Docker |
| Qdrant | Qdrant 1.13 in Docker |
| LanceDB | embedded (no service) |
| Chroma | embedded; service mode also profiled |
| Harness | Python 3.11+, `pytest-benchmark`, `prometheus-client` |
| Plots | matplotlib + seaborn |
| Reproducibility | `docker compose` with pinned versions, `make bench-all` |

## What this benchmark does NOT measure

- Filtered search (`WHERE category = 'X' AND vector ≈ q`) — Phase 6 stretch
- Hybrid search (BM25 + vector) — Phase 6 stretch
- Multi-tenancy at scale
- Geo-replicated reads
- Index recovery time after a crash
- GPU acceleration

These are real questions; they're omitted here on purpose to keep the comparison apples-to-apples.

## License

MIT. See [LICENSE](LICENSE).
