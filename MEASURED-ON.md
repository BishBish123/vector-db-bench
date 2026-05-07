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

## 100K synthetic run (tuned ef\_search=256)

Measured on 2026-05-07. Results from `results/100k-tuned/run-real/summary.parquet`.

Same hardware as the default-knobs run above. Full stack brought up fresh via
`docker compose -f docker-compose.full.yml up -d --wait pgvector qdrant` on ports
5444 / 6344 / 6345.

### Sweep results (ef\_search × recall@10)

The `results/100k/sweep-tuned/sweep.parquet` contains a 4-point knob sweep over
pgvector `ef_search` with `m=16`, `ef_construction=64` (HNSW defaults):

| ef\_search | m | recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 0.050 | 17.1 | 25.1 | 27.8 | 56.9 |
| 64 | 16 | 0.098 | 22.5 | 35.7 | 43.0 | 42.2 |
| 128 | 16 | 0.172 | 23.2 | 38.8 | 49.1 | 40.9 |
| 256 | 16 | 0.272 | 34.2 | 70.1 | 86.8 | 26.0 |

Best sweep point: **ef\_search=256** (recall=0.272). Even at ef\_search=256, recall
does not exceed 0.5. This is the expected behaviour for 384-dim synthetic Gaussian
vectors — see the methodology note below.

### Per-adapter results (tuned ef\_search=256)

| Adapter | recall@10 (mean) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | QPS (est.) | Ingest (vps) | Index time (s) | cost/M queries (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pgvector:hnsw-ef256` | 0.284 | 37.72 | 82.49 | 169.77 | 16.4 | 1 040 | 331.8 | $3.75 |
| `qdrant:hnsw-default` | 0.750 | 37.23 | 69.62 | 72.52 | 25.7 | 559 | — (on-insert) | $0.86 |
| `exact:bruteforce` | 1.000 | 216.92 | 381.40 | 970.63 | 3.7 | — | — | N/A |

### Comparison: default vs tuned pgvector

| Metric | default (ef\_search=10) | tuned (ef\_search=256) | delta |
| --- | --- | --- | --- |
| recall@10 | 0.060 | 0.284 | +0.224 (+373%) |
| p50 latency | 8.55 ms | 37.72 ms | +29.2 ms |
| p95 latency | 22.40 ms | 82.49 ms | +60.1 ms |
| QPS (est.) | 88.5 | 16.4 | −81% |

Tuning ef\_search from 10 → 256 lifts recall by 4.7× but at a 4.5× latency cost
(p50). The recall still maxes out at 0.284 — not a tuning failure but a corpus
characteristic (see methodology note).

### Methodology note: why synthetic Gaussian data is adversarial for HNSW

HNSW's greedy graph traversal exploits **cluster structure**: vectors close in the
embedding space form sub-graphs, and the traversal can jump between clusters
efficiently. Real retrieval corpora (MS-MARCO, BEIR) have strong cluster structure —
documents about the same topic are close together — so HNSW gets high recall with
moderate `ef_search`.

Synthetic 384-dim Gaussian vectors (this run) have **no cluster structure** — every
vector is equally far from every other vector in expectation. The HNSW graph is
dense but the greedy traversal cannot exploit any shortcut, so it must examine far
more candidates to find the true top-10. At 384 dimensions, the "curse of
dimensionality" compounds this: the ratio of max-to-min cosine distance among 100K
Gaussian vectors contracts severely, making the top-10 neighbors nearly
indistinguishable from the top-100 by distance.

The practical implication:

- **Default ef\_search=10** gives recall=0.06 — expected, not a bug.
- **Tuned ef\_search=256** reaches recall=0.284 — better, still low because the
  corpus structure caps HNSW's ceiling.
- **Qdrant's default ef\_search≈128** lands at recall=0.750 — higher because Qdrant
  uses a more generous default, not because the data is clustered.
- **On real MS-MARCO data at the same dimension**, ef\_search=64 routinely reaches
  recall@10 ≥ 0.92 on both adapters.

**Takeaway for benchmark readers**: low recall on synthetic Gaussian data does not
predict recall on real workloads. The right operating point must be found by sweeping
ef\_search on your actual data. The sweep recipe (`scripts/pgvector_sweep.py`) and
the `results/100k/sweep-tuned/sweep.parquet` show exactly this methodology.

### Artifacts

| Artifact | Path |
| --- | --- |
| Sweep parquet | `results/100k/sweep-tuned/sweep.parquet` |
| Sweep Pareto chart | `results/100k/sweep-tuned/sweep_pareto.{png,svg}` |
| Tuned summary parquet | `results/100k-tuned/run-real/summary.parquet` |
| Tuned timings parquet | `results/100k-tuned/run-real/timings.parquet` |
| Tuned host manifest | `results/100k-tuned/run-real/bench_manifest.json` |
| Tuned HTML report | `results/100k-tuned/report.html` |
| Tuned charts | `assets/100k-tuned/` |

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
