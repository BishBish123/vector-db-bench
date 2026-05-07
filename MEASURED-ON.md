# MEASURED-ON

Host metadata for the published benchmark numbers in this repository.
The `results/<name>/bench_manifest.json` for any committed run is the
machine-readable source of truth — this file restates the highlights in
prose so a reviewer doesn't have to grep the JSON.

The fields below are pulled from `results/demo/bench_manifest.json`
(the only run committed to the repo today), with two exceptions called
out inline. When `make bench-1m` lands its `results/full/` tree, this
file will be updated to cover that run alongside the demo.

## Demo run (`results/demo/`)

From the manifest's `host_metadata` + top-level fields:

- **Platform:** macOS-15.7.4-x86_64-i386-64bit
- **Machine:** x86_64, 8 logical cores
- **Total memory:** 8 GiB (8 589 934 592 bytes)
- **Python:** 3.12.13 (CPython)
- **Adapters:** `pgvector==0.4.2`, `qdrant-client==1.17.1`
- **Encoder:** `synthetic-gaussian` (dim=64, deterministic — no embedding
  model in this run; see `make bench-1m` for the `bge-small` MS-MARCO
  encoding)

Not captured by the manifest schema today; sourced from the repo:

- **Containers:** `pgvector/pgvector:0.8.0-pg17` and `qdrant/qdrant:v1.17.0`
  (from `docker-compose.yml`)
- **vdbbench version:** 0.1.0 (from `pyproject.toml`)

The latency numbers in `README.md` are dominated by host noise at this
scale (5 000 vectors, HNSW defaults). The Pareto curves that actually
discriminate the adapters live further up the corpus-size axis — see
the `make bench-1m` recipe in `README.md`.

## 100K synthetic run (`results/100k/run-real/`)

Measured on 2026-05-06. All numbers sourced from
`results/100k/run-real/bench_manifest.json` and `summary.parquet`.

### Host metadata

- **Platform:** macOS-15.7.4-x86_64-i386-64bit
- **Machine:** x86_64, 8 logical cores
- **Total memory:** 8 GiB (8 589 934 592 bytes)
- **Python:** 3.12.13 (CPython)
- **Adapters:** `pgvector==0.4.2`, `qdrant-client==1.17.1`, `exact` (in-process brute-force)
- **Encoder:** `synthetic-gaussian` (dim=384, deterministic)
- **Containers:** `pgvector/pgvector:0.8.0-pg17` (shm-size=1g), `qdrant/qdrant:v1.17.0`
- **vdbbench version:** 0.1.0
- **Run date:** 2026-05-06T23:18:42Z → 2026-05-06T23:29:36Z (~11 minutes total)
- **Corpus:** 100 000 vectors × 384 dim, 100 queries, k=10, profile=warm

### Per-adapter results

| Adapter | recall@10 (mean) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | QPS (est.) | Ingest (vps) | Index time (s) | cost/M queries (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pgvector:hnsw-default` | 0.060 | 8.55 | 22.40 | 25.88 | 88.5 | 1 295 | 366.9 | $0.70 |
| `qdrant:hnsw-default` | 0.778 | 53.57 | 87.58 | 103.74 | 18.1 | 634 | — (on-insert) | $1.23 |
| `exact:bruteforce` | 1.000 | 154.49 | 221.59 | 305.93 | 6.0 | 151 208 119 | — | N/A |

**Notes:**
- `pgvector` recall@10 = 0.06 on a 384-dim synthetic-gaussian corpus with HNSW defaults (m=16,
  ef_construction=64). HNSW search uses ef=10 by default, which is very tight relative to k=10 on
  high-dimensional random data — low recall is expected here and matches the demo-run pattern.
  Recall improves with higher ef_search (see sweep recipe).
- `qdrant` recall@10 = 0.778 with its default HNSW params (m=16, ef=128 at index time). Qdrant
  configures ef_search more generously by default, hence the higher recall at the cost of higher
  latency.
- `exact` is the ground-truth baseline (brute-force cosine, recall=1.0 by definition).
- pgvector required `--shm-size=1g` (Docker default 64 MB is insufficient for 100K-vector HNSW
  index build; `DiskFull` on `/dev/shm` otherwise).
- Ports: pgvector on 5444, qdrant on 6344/6345 (default 5433/6333 were occupied by dev containers).

### Artifacts

| Artifact | Path |
| --- | --- |
| Summary parquet | `results/100k/run-real/summary.parquet` |
| Timings parquet | `results/100k/run-real/timings.parquet` |
| Host manifest | `results/100k/run-real/bench_manifest.json` |
| HTML report | `results/100k/report.html` |
| Charts | `assets/100k/` |

---

## Full run (1M MS-MARCO) — NOT YET MEASURED

The canonical 1 M-vector MS-MARCO sweep has not yet been executed on
publication-grade hardware.  When the run is published, this section
will quote the same `host_metadata` fields sourced from
`results/1m/run-<YYYYMMDD>/bench_manifest.json`.

### How to reproduce

Run the automation script on a Linux host with ≥50 GB free disk and
≥16 GB RAM (8+ cores recommended):

```bash
# First time — downloads MS-MARCO (~8 GB) and encodes 1M passages
./scripts/run_1m_bench.sh

# If data/encoded-1m already exists from a previous prep run
./scripts/run_1m_bench.sh --skip-prep

# Dry-run: print every step without executing
./scripts/run_1m_bench.sh --dry-run
```

Or via Make:

```bash
make bench-1m-real
```

### Where results land

| Artifact | Path |
| --- | --- |
| Summary parquet | `results/1m/run-<YYYYMMDD>/summary.parquet` |
| Timings parquet | `results/1m/run-<YYYYMMDD>/timings.parquet` |
| Host manifest | `results/1m/run-<YYYYMMDD>/bench_manifest.json` |
| HTML report | `results/1m/run-<YYYYMMDD>/report.html` |
| Charts | `assets/1m/` |

Copy the manifest fields into the table below once the run completes:

| Field | Value |
| --- | --- |
| Platform | _TODO_ |
| Machine | _TODO_ |
| Total memory | _TODO_ |
| Python | _TODO_ |
| Adapters | _TODO_ |
| Encoder | `BAAI/bge-small-en-v1.5` (384-dim, bge-small) |
| vdbbench version | _TODO_ |
| Run date | _TODO_ |
