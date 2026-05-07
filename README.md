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
2. Run `make bench-demo` on a laptop and reproduce the **demo** numbers (and `make bench-100k` / `make bench-1m` for larger sweeps)
3. Re-run the analysis themselves from the published parquet

That's the bar.

> **Demo vs full.** The numbers in this README and the parquet checked into `results/demo/` come from a 5 000-vector synthetic sanity sweep — small enough to ship in the repo and re-run on a laptop in seconds. The canonical full sweep is a 1 M-vector MS-MARCO run produced by `make bench-1m` and persisted to `results/full/summary.parquet` (not committed; the recipe is below). When the README cites "Pareto frontier" or "p95 latency", it means the demo unless explicitly tagged `[full]`.

> **Adapter coverage of the committed demo parquet.** The `results/demo/summary.parquet` checked in here is the **2-adapter** sweep (pgvector + qdrant) captured on the published bench host — an Intel macOS laptop where `lancedb` and `chromadb` have no wheels. A reviewer running `make bench-demo` on Linux / Apple Silicon / WSL2 regenerates a **4-row** parquet with all four adapters; the chart will look different (an extra two points on the Pareto frontier) and that is expected, not a regression. See the Platform support table below for the wheel availability matrix.

## Headline chart (5K-vector demo)

The numbers below come from a 5 000-vector synthetic corpus with brute-force ground truth, run on a 2020 Intel MacBook Air against pgvector pg17 + qdrant 1.17 in Docker. They're a sanity-check of the pipeline, not the canonical benchmark — `make bench-1m` is what produces the full 1 M-vector MS-MARCO sweep.

![Pareto frontier — recall vs p95 latency](assets/pareto.png)

| DB | Ingest (vps) | p95 latency (ms) | Recall@10 | NDCG@10 | QPS (est.) | $/M-queries (est.) | Pg total relation size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector (HNSW defaults) | 3 016 | 9 | 0.876 | 0.920 | ~150 | ~$0.41 | 4.4 MB |
| qdrant (HNSW defaults) | 4 814 | 14 | 1.000 | 1.000 | ~88 | ~$0.70 (unverified) | (mem) |

These numbers come from the published bench host (see `results/demo/bench_manifest.json` for CPU / OS / interpreter); re-running `make bench-demo` overwrites `results/demo/summary.parquet` with your host's numbers — the chart updates, but this README table does not. The "Pg total relation size" column is `pg_total_relation_size` (table + index) for pgvector and 0 for qdrant (HNSW lives in memory). The "$/M-queries" column is a compute-time × public price-list estimate (as of 2026-05); see [§ Cost methodology](#cost-methodology) below.

Numbers regenerated from `results/demo/summary.parquet` against the current code (pgvector connection-reuse + RSS sampling fixes). Earlier drafts of this README quoted pgvector p95 ~57 ms with a "per-query connection overhead" caveat — that caveat is gone because the adapter now reuses a single psycopg connection across the whole `BenchSpec` lifecycle (see [BLOG.md](BLOG.md) for the methodology story).

Notes worth flagging — these are *exactly* the kinds of caveats the blog post will dig into:

- 5 000 vectors is well below where ANN-vs-exact differences matter. The pgvector vs Qdrant ordering you see here is dominated by per-query overhead and host noise rather than search-algorithm quality. The interesting curves come from sweeping `ef_search` over a real corpus at scale; treat these numbers as a smoke test of the harness, not a verdict.
- 100 % recall on Qdrant at this scale is expected — HNSW with default `m=16` over 5 000 vectors is essentially exact. pgvector's 0.876 is also fine (HNSW with no `ef_search` tuning).

## Reproduce

### Prerequisites

- `uv` (the package manager — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `docker` + `docker compose` v2 (modern subcommand, not the legacy `docker-compose`)
- `bash` (the Makefile uses bash; macOS / Linux / WSL2 only)

### Re-plot only (no Docker required)

The committed `results/demo/summary.parquet` is enough to regenerate every
chart on its own:

```bash
uv run vdbbench plot --summary results/demo/summary.parquet --out assets
```

## HTML report

A self-contained, browser-ready HTML report is committed at
[`results/demo/report.html`](results/demo/report.html). It embeds the
methodology section, headline metrics table, all charts as inline `data:` URIs,
and per-adapter detail tables — everything a reviewer needs without scrolling
through markdown.

To regenerate it:

```bash
make report
# or directly:
uv run vdbbench report html --from results/demo/ --out results/demo/report.html --charts-dir assets
```

The `--charts-dir` flag is optional; omit it to produce a no-charts HTML
that is still fully self-contained.

### Demo (the numbers in this README)

```bash
git clone https://github.com/BishBish123/vector-db-bench.git
cd vector-db-bench

# 1. Install (uv-managed, all extras the platform supports).
make install

# 1b. (Optional) End-to-end offline smoke — no Docker, no model download.
#     Exercises corpus → encode → bench → plot in a few seconds against
#     the in-package brute-force adapter. Useful as a "did the install
#     actually work" check before pulling the bench images.
make smoke

# 2. Bring up pgvector + qdrant in Docker.
#    `make up` pulls images first, then waits for the containers to report healthy.
#    If host port 5433 / 6333 is already allocated, override via env:
#      make up PGVECTOR_PORT=5444 QDRANT_PORT=6400 QDRANT_GRPC_PORT=6401
#    The same vars are honoured by docker-compose.yml *and* by `make
#    bench` / `make bench-demo` / `make bench-1m` — both legs read the
#    same PGVECTOR_PORT / QDRANT_PORT so a rebound port reaches the
#    bench CLI:
#      make bench-demo PGVECTOR_PORT=5444 QDRANT_PORT=6400 QDRANT_GRPC_PORT=6401
#    Override PGVECTOR_DSN / QDRANT_URL directly to point the bench at
#    a non-Docker service.
make up

# 3. Demo run — 5 000 synthetic vectors, ~30 seconds end-to-end.
#    `make bench-demo` is the one-command wrapper for exactly the steps below.
#    Prefer `make bench-demo` over the by-hand commands when you've
#    overridden PGVECTOR_PORT / QDRANT_PORT — the make target threads
#    those vars into the bench DSN/URL automatically; the manual
#    commands hardcode the defaults.
make bench-demo
# ...or run them by hand (uses the env-var defaults; if you've rebound
# either container port, swap in $PGVECTOR_PORT / $QDRANT_PORT):
uv run vdbbench prep   --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
uv run vdbbench bench  --encoded data/encoded-demo --out results/demo \
                       --pgvector-dsn postgresql://bench:bench@localhost:${PGVECTOR_PORT:-5433}/bench \
                       --qdrant-url http://localhost:${QDRANT_PORT:-6333}
uv run vdbbench plot   --summary results/demo/summary.parquet --out assets
```

For a larger sweep on the same laptop, `make bench-100k` runs the pipeline at 100 000 *synthetic* vectors (configurable via `SAMPLE_SIZE`) and writes to `results/100k/` — it's a scale-up smoke test for the harness, not an MS-MARCO run. Only `make bench-1m` switches the corpus to MS-MARCO; that is the canonical full sweep below.

`results/demo/summary.parquet`, the matching `timings.parquet`, and the `bench_manifest.json` from the run that produced them are checked into the repo — a reviewer can run just the [Re-plot only](#re-plot-only-no-docker-required) path above and inspect the published numbers without bringing services up. The manifest captures encoder identity, adapter versions, encoded-bundle fingerprint, and host metadata so the parquet is auditable, not just present.

### Full (1 M MS-MARCO, the canonical benchmark)

The full sweep is reproducible but **not** checked in — the parquet would be too large and the recall numbers are tied to the host the run was performed on. The recipe:

```bash
uv run vdbbench prep  --dataset msmarco --sample-size 1000000 --out data/encoded-1m
uv run vdbbench bench --encoded data/encoded-1m --out results/full --all --profile p99
uv run vdbbench plot  --summary results/full/summary.parquet --out assets/full
```

`make bench-1m` is the one-command wrapper (defined in the Makefile; `prep --dataset msmarco --sample-size 1000000`, then `bench --all --profile p99`, then `plot`). It runs each adapter once with HNSW defaults at the `p99` profile (50 warm-up queries + 5 measured repeats) — not yet a Pareto sweep over `ef_search` / `probes` / `nprobes`; that knob-grid runner is future work (see Methodology below). Wall-clock is dominated by the MS-MARCO encode (~8 GB of HF data + ~400 MB of model weights to download on first run) and the per-adapter ingest + index build over 1 M vectors. Persist the resulting `results/full/` tree (parquet + `bench_manifest.json`) when you publish numbers.

The `--lancedb-path` and `--chroma-path` flags (used by `--all`) are skipped on Intel macOS (no wheels for `lancedb` / `chromadb`+`onnxruntime`); run on Linux / arm64 macOS / WSL2 for the full four-way comparison.

## Methodology

- **Corpora.** MS-MARCO via [BeIR/msmarco](https://huggingface.co/datasets/BeIR/msmarco) (sample-size capped, all judged passages always kept), or any other BEIR dataset (`scifact`, `nfcorpus`, `fiqa`, …). A pure-Python synthetic corpus with brute-force ground truth is included for CI and smoke tests.
- **Embedding model.** `BAAI/bge-small-en-v1.5` (384-dim) by default; pluggable via `--embed-model`.
- **Sample size.** `--sample-size` defaults to 5 000 across `vdbbench prep` (matches the committed `results/demo/`). `make bench-100k` and `make bench-1m` override explicitly to 100 000 / 1 000 000.
- **Hardware.** Single machine, documented per run. No GPU unless the run says so.
- **Fairness.** Per-spec warm-up queries (default 10) are discarded before timing; `repeats > 1` runs the full query set multiple passes.
- **Repeats.** Per-(db, params) summary aggregates across `repeats × n_queries` measured timings.
- **Recall denominator.** The `relevance > 0` qrels are the positives; judged-negative entries (`relevance == 0`) are explicitly excluded from recall and NDCG.

The full sweep of HNSW `ef_search`, IVF `lists`/`probes`, and IVF-PQ `num_partitions` is future work: the harness already plumbs `params` straight to the adapters, so a knob-grid runner (driving `run_bench` over a list of `BenchSpec` values) is the missing piece. Today the supported workflow is to invoke `run_bench` directly from a Python script with the per-knob specs you want to compare; the CLI exposes only the four standalone commands listed in `vdbbench --help` (`prep`, `bench`, `plot`, `version`).

## Cost methodology

The `$/M-queries` column in the headline table and in `summary.parquet` (`cost_per_million_queries_usd`) is a **compute-time × public price-list estimate** — not a production-measured bill.

**Formula:**

```
qps                    = total_queries / total_query_seconds   # measured by the bench
hourly_query_capacity  = qps * 3600
cost_per_query         = compute_per_hour_usd / hourly_query_capacity
                         + per_query_usd                       # metered charge, if any
cost_per_million       = cost_per_query * 1_000_000
```

**Price reference** (see `src/vdbbench/pricing.py` for the full table with source URLs):

| Adapter | Tier | $/hr compute | $/query | Verified? | As of |
| --- | --- | ---: | ---: | --- | --- |
| pgvector | Neon Scale — 1 CU | $0.222 | $0 | Yes | 2026-05 |
| qdrant | Qdrant Cloud Standard (smallest) | $0.08 | $0 | No — see note | 2025-01 |
| chroma | Chroma Cloud serverless | $0 | ~$1.1e-8 | Yes | 2026-05 |
| lancedb | Embedded + S3 Standard | $0 | $0 | No — see note | 2025-01 |
| exact | In-process baseline | $0 | $0 | Yes | — |

Notes on unverified entries:
- **qdrant**: Qdrant Cloud's live pricing page (qdrant.tech/pricing) requires the interactive calculator; the $0.08/hr figure is from earlier public documentation and is marked unverified in `pricing.py`.
- **lancedb**: Self-hosted embedded — compute cost is effectively $0. S3 storage ($0.023/GB-month) is a separate line item not reflected in the per-query cost; the figure could not be confirmed from the live AWS pricing page in this session.

These numbers are **compute-time × public price-list estimates** as of the dates shown. They are not production-measured. The prices assume the smallest plausible production-grade tier for each cloud service.

## Profiling

vdbbench ships three profiling integrations that help you understand *why* one adapter is slower than another.

### Memory tracking (default on)

Every bench run captures per-adapter RSS via a background thread sampler (`MemorySampler`) at 100 ms intervals across the ingest + index + query phases.  The results land in `summary.parquet`:

| Column | Meaning |
| --- | --- |
| `baseline_rss_bytes` | Process RSS before any adapter work (Python + NumPy interpreter overhead) |
| `index_rss_bytes` | RSS delta (baseline-subtracted) after `build_index()` returns |
| `peak_rss_bytes` | Max baseline-subtracted RSS across all phases (catches transient spikes) |
| `adapter_memory_bytes` | Peak RSS sampled by the background thread across ingest + query phases |

The `analysis.ipynb` memory chart uses `adapter_memory_bytes` (background-sampled peak) to draw the per-adapter bar chart.  This requires a standard `vdbbench bench` run — memory tracking is always on by default, no extra flag needed.

### py-spy flame graphs (opt-in)

Install the extra and then pass `--profiler py-spy`:

```bash
pip install "vdbbench[profile]"      # installs py-spy>=0.4

# One-liner profiling run against the smoke data:
make profile-demo                    # → results/profile-demo/profile.svg

# Or via CLI directly:
uv run vdbbench bench \
    --adapter exact \
    --encoded smoke-out/encoded \
    --profiler py-spy \
    --out results/profile-demo
```

`py-spy record` attaches to the bench process and writes a SVG flame graph to `results/profile-demo/profile.svg`.  Open the SVG in any browser to explore the call stack.

**Soft-fail**: if `py-spy` is not on PATH the bench still runs — it emits a `UserWarning` and skips the flame graph, so CI passes cleanly without the extra installed.

**Linux note**: `py-spy record` may need elevated privileges on kernels with `ptrace_scope=1` (common default).  Run with `sudo` or add `SYS_PTRACE` to the Docker container.

### iostat I/O stats (opt-in)

`iostat` ships with `sysstat` on Linux and is part of the base system on macOS.  Pass `--profiler iostat` to record CPU + disk I/O stats alongside the bench:

```bash
uv run vdbbench bench \
    --adapter exact \
    --encoded smoke-out/encoded \
    --profiler iostat \
    --out results/profile-demo
```

Output lands in `results/profile-demo/iostat.txt`.  **Recommended** when you suspect I/O saturation (high `await`, low `%util` headroom) is limiting throughput.

**Soft-fail**: if `iostat` is not on PATH the bench still runs with a `UserWarning`.

### ps_mem per-process memory breakdown (opt-in)

`ps_mem` reports the detailed memory split — private, shared, and swap — for the bench process.  Unlike the always-on `MemorySampler` (which tracks RSS), `ps_mem` shows how much memory is *truly exclusive* to the process versus shared with other processes (e.g. shared libraries).

Install the system binary first:

```bash
# macOS
brew install ps_mem

# Debian/Ubuntu
sudo apt install ps_mem
```

Then run the bench with `--profiler ps_mem`:

```bash
# One-liner profiling run:
make profile-demo-mem               # → results/profile-demo-mem/ps_mem.log

# Or via CLI directly:
uv run vdbbench bench \
    --adapter exact \
    --encoded smoke-out/encoded \
    --profiler ps_mem \
    --out results/profile-demo-mem
```

Output lands in `results/profile-demo-mem/ps_mem.log` as a series of timestamped snapshots (one every 5 seconds by default).  Each snapshot contains a `Private + Shared = RAM used` table for the bench process.

**Recommended** when `peak_rss_bytes` in the summary parquet is high and you want to confirm whether the memory is truly private to the bench process or mostly shared pages (shared libraries, mmap'd data files).  Complements `MemorySampler` (in-process RSS) and works well alongside `--profiler py-spy` for a complete picture: py-spy shows *where* memory allocations happen; ps_mem shows *how much* is exclusive to the process.

**Soft-fail**: if `ps_mem` is not on PATH the bench still runs with a `UserWarning`.

## Prometheus metrics

Every `vdbbench bench` run emits Prometheus metrics in-process. To expose them over HTTP for scraping:

```bash
uv run vdbbench bench --adapter exact --dataset synthetic --prometheus-port 9100 &
curl -s localhost:9100/metrics | grep vdbbench
```

Metrics exported:

| Metric | Type | Description |
| --- | --- | --- |
| `vdbbench_ingest_vectors_total` | Counter | Vectors ingested per adapter run |
| `vdbbench_query_latency_seconds` | Histogram | Per-query latency (buckets: 1ms–1s) |
| `vdbbench_query_recall_at_k` | Gauge | Mean recall@k from the last spec |
| `vdbbench_bench_duration_seconds` | Summary | Full spec wall-clock time |

All metrics carry an `adapter` label. `vdbbench_query_recall_at_k` also carries a `k` label.

## Stack

| Layer | Choice |
| --- | --- |
| Embeddings | `sentence-transformers` (bge-small, nomic) |
| Corpus | BEIR via Hugging Face `datasets` (streaming + judged-aware sampling) |
| pgvector | Postgres 17 + `pgvector` 0.8.0 in Docker (image tag `pgvector/pgvector:0.8.0-pg17`) |
| Qdrant | Qdrant 1.17 in Docker |
| LanceDB | embedded (no service) |
| Chroma | embedded persistent client |
| Harness | Python 3.11+, custom `run_bench()` runner |
| Plots | matplotlib |
| Reproducibility | `docker compose` with pinned versions, `make` targets, parquet outputs |

## Encoders

Two embedding models are registered out of the box:

| Key | HuggingFace ID | Dim | `trust_remote_code` |
| --- | --- | ---: | --- |
| `bge` (default) | `BAAI/bge-small-en-v1.5` | 384 | No |
| `nomic` | `nomic-ai/nomic-embed-text-v1.5` | 768 | **Yes** |

### Selecting an encoder

**Makefile (recommended):**

```bash
make bench-demo                   # bge (384-dim, default)
make bench-demo ENCODER=nomic     # nomic (768-dim)
make bench-demo-nomic             # convenience alias for the above
```

**CLI directly — short name (recommended):**

```bash
uv run vdbbench prep --dataset synthetic --encoder bge    # bge (384-dim, default)
uv run vdbbench prep --dataset synthetic --encoder nomic  # nomic (768-dim, uses trust_remote_code)
```

**CLI directly — raw HuggingFace ID:**

```bash
uv run vdbbench prep --embed-model BAAI/bge-small-en-v1.5 --out data/encoded-bge
uv run vdbbench prep --embed-model nomic-ai/nomic-embed-text-v1.5 --out data/encoded-nomic
```

> **Note:** `--encoder nomic` automatically passes `trust_remote_code=True` to SentenceTransformer. Using `--embed-model nomic-ai/nomic-embed-text-v1.5` directly does **not** set `trust_remote_code` and will fail to load the nomic model. Prefer `--encoder nomic` for nomic-embed-text-v1.5.

**Python API:**

```python
from vdbbench.embed.registry import build_encoder, ENCODER_REGISTRY

enc = build_encoder("bge")    # SentenceTransformerEncoder, dim=384
enc = build_encoder("nomic")  # SentenceTransformerEncoder, dim=768
```

### Expected impact of the dim difference

| Dimension | Memory per 1 M vectors | Typical recall@10 | Typical p95 latency |
| ---: | --- | --- | --- |
| 384 (bge) | ~1.5 GB float32 | Good (MTEB ~51) | Lower |
| 768 (nomic) | ~3 GB float32 | Better (MTEB ~62) | Higher |

Higher dimensions improve recall at the cost of more index memory and slightly
higher query latency.  The difference matters most for large corpora; at the
5 000-vector demo scale it is dominated by noise.

### `trust_remote_code` security note for nomic

`nomic-embed-text-v1.5` ships a custom `modeling_hf_nomic_bert.py` that
`sentence-transformers` must execute from the HuggingFace cache when
`trust_remote_code=True` is passed.  This is the same flag used by other
community models (e.g. Falcon, InternLM).  **Before using the nomic encoder
in a production or shared environment, read the model card at
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 and audit the
modeling file.**

Note: from `transformers >= 5.5.0` and `sentence-transformers >= 5.3.0`,
`trust_remote_code` will no longer be required for this model.

## Encoder comparison

`vdbbench compare-encoders` benches two (or more) encoders against the same
corpus and adapter, then writes a side-by-side parquet + chart so you can see
the recall / latency / dimension tradeoff without running two full `prep +
bench + plot` pipelines by hand.

**Quick smoke run (offline, no model download):**

```bash
make compare-encoders              # bge only — fast, no download
```

This writes `results/encoder-compare/encoder_comparison.parquet` and
`results/encoder-compare/encoder_comparison.png` using bge (384-dim) against
the in-process exact adapter.  The committed parquet under
`results/encoder-compare/` was produced by this command on an Intel macOS
host where `sentence-transformers` is not available; it is bge-only.

**Full bge-vs-nomic comparison (requires model download):**

```bash
make compare-encoders ENCODERS=bge,nomic   # ~500 MB nomic download on first run
```

Or via the CLI directly:

```bash
uv run vdbbench compare-encoders \
    --adapter exact \
    --encoders bge,nomic \
    --dataset synthetic \
    --out results/encoder-compare
```

The resulting chart plots recall@k on the left y-axis and p95 latency on the
right, one bar pair per encoder, with the embedding dimension annotated above
each pair.

> **Note:** `nomic` requires `trust_remote_code=True` — read the security note
> in the Encoders section above before running with nomic in a production
> environment.

## Platform support

| Platform | Local dev (`make install`) | Local bench |
| --- | --- | --- |
| Linux x86_64 | ✅ all extras | ✅ |
| macOS arm64 (Apple Silicon) | ✅ all extras | ✅ |
| macOS x86_64 (Intel) | ✅ core + dev only (`make install-min`) | ⚠️ pgvector + qdrant only — lancedb / chroma require Linux / arm64 macOS / WSL2 (no wheels for Intel macOS) |

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
    ADR-005-...           schema-version every persisted manifest
```

See [BLOG.md](BLOG.md) for the writeup of one specific tradeoff this bench surfaced, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layered design, and the [docs/adr/](docs/adr/) directory for the design decisions.

## Knob-grid Pareto sweep

The `sweep` command runs every Cartesian product of adapter knobs against the same encoded dataset and plots the recall-vs-latency Pareto frontier.

```bash
# Offline smoke — exact adapter, 6 trials (2 metrics x 3 k_neighbors values)
make sweep

# Larger grid against your own encoded bundle (no Docker for exact adapter)
uv run vdbbench sweep \
    --adapter exact \
    --grid metric=cosine,l2 \
    --grid k_neighbors=4,8,16,32 \
    --encoded smoke-out/encoded \
    --out results/sweep

# Regenerate the chart from a previous sweep parquet
uv run vdbbench plot-sweep --sweep-parquet results/sweep/sweep.parquet --out results/sweep
```

The sweep writes two artifacts to `--out`:
- `sweep.parquet` — per-trial metrics (recall@k, qps, p50/p95/p99 latency, ingest seconds)
- `sweep_pareto.{png,svg}` — scatter plot with Pareto-optimal points filled and connected

![Knob-grid Pareto sweep](results/sweep/sweep_pareto.png)

**Pareto frontier algorithm:** A result is dominated when some other result has both higher-or-equal recall *and* lower-or-equal p95 latency, with at least one inequality strict. Only non-dominated points are highlighted on the chart; the rest are rendered as open markers. Ties (same recall and same p95) are both kept.

**Python API:**

```python
from vdbbench.sweep import SweepSpec, run_sweep, pareto_frontier
import asyncio

spec = SweepSpec(
    adapter="exact",
    parameter_grid={"metric": ["cosine", "l2"], "k_neighbors": [4, 8, 16]},
    dataset="smoke-out/encoded",
    top_k=10,
)
results = asyncio.run(run_sweep(spec))
frontier = pareto_frontier(results)
```

## Wikipedia corpus

The benchmark supports a second corpus loader for Wikipedia alongside MS-MARCO.

### Environment requirements

Wikipedia streaming requires the `datasets` library.  It is listed under the
`[wiki]` optional-dependency group so it does not pull into the core install on
platforms where it would be heavy:

```bash
uv sync --extra wiki
```

### Loading the full corpus

Wikipedia is large (~20 GB compressed).  You must pass `--limit` to cap the
article count before the download starts:

```bash
uv run vdbbench prep \
  --dataset wikipedia \
  --limit 10000 \
  --out data/encoded-wiki
```

Without `--limit` the command errors out with a clean message rather than
silently starting a multi-hour download.

### Offline / fixture mode

For tests and smoke runs the loader accepts a JSONL fixture file instead of
streaming from HuggingFace.  Each line must be
`{"id": ..., "title": ..., "text": ...}`:

```bash
uv run vdbbench prep \
  --dataset wikipedia \
  --fixtures evals/wikipedia_fixtures.jsonl \
  --limit 5 \
  --out smoke-out-wiki/encoded
```

The committed file `evals/wikipedia_fixtures.jsonl` contains 8 hand-curated
Wikipedia-style rows and is the fixture used by the test suite.

### Chunker behaviour

Wikipedia articles can be many kilobytes, so the loader splits each article
into overlapping chunks before embedding:

- Articles are split on `\n\n` (paragraph boundaries) first.
- Paragraphs longer than `--wiki-chunk-size` (default 512 chars) are
  hard-split at that boundary.
- A sliding window with `--wiki-overlap` (default 64 chars) of overlap
  carries the tail of each chunk into the next for context continuity.
- The chunk size contract is strict: no emitted chunk exceeds `chunk_size`
  characters even after the overlap prefix is applied.

Each chunk is identified as `<doc_id>_<chunk_index>` and carries the parent
article's title for display and re-ranking.

### Ground-truth note

Wikipedia has no bundled query/qrel ground truth.  The `prep` command
synthesises one self-relevant query per 1 000 chunks (or at least one) using
the chunk's title as the query text, with `relevance=1.0`.  This is sufficient
for smoke-testing the pipeline and the `exact` adapter, but is **not** a valid
recall benchmark.  Bring your own qrels if you need Wikipedia retrieval metrics.

## Running the tests

```bash
make test           # unit tests only — `not integration and not slow`
make test-integration   # requires Docker (skipped without containers up)
make test-all       # unit + integration + slow (the full suite)
```

`make test` is the canonical local invocation — it filters out `slow`
and `integration` markers so a clean run finishes in seconds rather
than the minutes a full `uv run pytest` would take. Use `make test-all`
before publishing benchmark numbers; CI runs the same matrix.

## Analysis notebook

`analysis.ipynb` at the repo root regenerates every chart from the committed
`results/demo/summary.parquet` — no Docker, no running services required.

### Run it

```bash
make analysis          # executes notebook, writes analysis.executed.ipynb + assets/
```

Or open it interactively:

```bash
uv run jupyter lab analysis.ipynb
```

### What it produces

| Chart | File |
| --- | --- |
| Cost vs latency Pareto | `assets/analysis-cost-pareto.png` |
| Recall vs latency Pareto | `assets/analysis-recall-pareto.png` |
| Ingest throughput | `assets/analysis-ingest.png` |
| Memory footprint | `assets/analysis-memory.png` (if data present) |

The notebook handles schema evolution gracefully — columns absent from older
parquets are derived or skipped with a printed note rather than crashing.

### Install notebook deps

The `[plot]` extras group adds `matplotlib`, `jupyter`, and `nbclient` without
touching the base install:

```bash
uv sync --extra plot
```

### Keeping notebook in sync with the harness

`tests/test_notebook.py` (marked `slow`) executes the notebook via `nbclient`
and asserts that no cells errored and that the expected PNGs were written.
`make test-all` runs it; `make test` (unit-only) skips it.

## Reproducibility

The benchmark is containerised so anyone can reproduce the smoke numbers exactly,
regardless of their local Python environment.

```bash
# One-command verification: build image → run bench → diff against baseline
./scripts/reproduce.sh
```

### Full 1M MS-MARCO run (press-one-button)

For the canonical 1 M-vector MS-MARCO sweep, use `scripts/run_1m_bench.sh`:

```bash
# Full pipeline: prereq checks → docker up → prep → bench → report → plot → docker down
./scripts/run_1m_bench.sh

# If data/encoded-1m already exists (skip the 2-4 h encode step)
./scripts/run_1m_bench.sh --skip-prep

# Dry-run: print every step without executing
./scripts/run_1m_bench.sh --dry-run

# Or via Make
make bench-1m-real
```

**Hardware requirements for the 1M run:**

| Resource | Minimum | Recommended |
| --- | --- | --- |
| Disk (free) | 50 GB | 100 GB |
| RAM | 16 GB | 32 GB |
| CPU cores | 4 | 8 |
| OS | Linux x86_64 or arm64 | Ubuntu 22.04+ |

**Estimated wall-clock** (shared-cpu-1x, 4 cores): 4–8 h total
- MS-MARCO download + encode: 2–4 h (HF bandwidth + CPU bound)
- pgvector ingest + index: 1–2 h
- qdrant / lancedb / chroma: 1–2 h combined

Results land in `results/1m/run-<YYYYMMDD>/` (parquet + HTML report + manifest).
Charts land in `assets/1m/`.  See `MEASURED-ON.md` for the placeholder that
will be filled in when the first publishable run completes.

What the script does:

1. Builds the `vdbbench` Docker image (multi-stage; python:3.12-slim + uv).
2. Starts the full stack (`pgvector` + `qdrant` + `vdbbench`) via
   `docker-compose.full.yml`.
3. Runs the bench inside the container with the same 5 000-vector synthetic
   corpus used to produce `results/demo/summary.parquet`.
4. Copies results to `./results/reproduced/`.
5. Diffs the produced `summary.parquet` against `results/demo/summary.parquet`
   (the committed baseline); exits non-zero if p95 latency drifts > 50 % or
   recall@10 drifts > 0.05 absolute.

Tolerances are adjustable via env vars:

```bash
TOLERANCE_LATENCY_PCT=30 TOLERANCE_RECALL_ABS=0.02 ./scripts/reproduce.sh
```

For offline verification (no Docker required — uses the in-process `exact` adapter):

```bash
./scripts/reproduce.sh --smoke-only
```

### Make targets

| Target | What it does |
| --- | --- |
| `make docker-build` | Build the `vdbbench:latest` image |
| `make docker-run` | Bring up full stack via `docker-compose.full.yml` |
| `make reproduce` | Full reproducibility check (build + run + diff) |

The weekly CI workflow at `.github/workflows/reproducibility.yml` runs
`scripts/reproduce.sh --smoke-only` automatically on every push to `main` and
every Sunday at 02:00 UTC, surfacing drift before it reaches users.

## License

MIT. See [LICENSE](LICENSE).
