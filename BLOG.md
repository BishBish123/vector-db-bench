# What a 5K-vector smoke run already tells you about pgvector vs Qdrant

> *Companion blog post for [vector-db-bench](https://github.com/BishBish123/vector-db-bench). The numbers cited here come from the **demo** run (5 000 synthetic vectors, checked into `results/demo/summary.parquet`); the canonical 1 M-vector MS-MARCO sweep is what `make bench-1m` produces against `results/full/`. This post is about what the demo already reveals, and the methodology pitfalls I found writing it.*

I built a benchmark harness because every "which vector DB should I use?" article I read was either a vendor blog post or a 100-line micro-bench whose corpus, hardware, and tuning weren't documented. The repo is the harness; this post is the story of one specific tradeoff I saw on the very first run.

## The setup

* **Corpus:** 5 000 synthetic L2-normalized vectors at dim 64 with brute-force ground truth (qrels are top-10 of cosine similarity over the same vectors). This is the demo run that ships in `results/demo/`, not the canonical benchmark — for the 1 M MS-MARCO numbers run `make bench-1m`.
* **Hardware:** 2020 Intel MacBook Air, 4 GB allotted to colima Docker.
* **Adapters:** `pgvector/pgvector:pg17` and `qdrant/qdrant:v1.17.0`, both via the same `BenchSpec(...)` lifecycle (`setup → ingest → build_index → warm-up → search × repeats → teardown`).
* **Knobs:** HNSW defaults on both. No `ef_search` sweep yet.

## The numbers

| DB | Ingest (vps) | p95 latency | Recall@10 | QPS (est.) |
| --- | ---: | ---: | ---: | ---: |
| pgvector | 3 016 | 9 ms | 0.876 | ~150 |
| qdrant | 4 814 | 14 ms | 1.000 | ~88 |

Headlines write themselves: *"pgvector is faster on p95 and Qdrant has perfect recall."* That's still not the headline — keep reading.

> **Methodology updates (May 2026).** These numbers reflect the round-1 + round-2 + round-3 fixes shipped in this repo, regenerated against `results/demo/summary.parquet` on the same host the rest of the post cites. The earlier draft quoted pgvector p95 ~57 ms because the harness was opening a new psycopg connection per query; the table now uses the connection-reuse fix. The earlier draft also said RSS sampling was a follow-up; it's now part of every run. See the [commit history](https://github.com/BishBish123/vector-db-bench/commits/main) for the exact changes that affect these numbers.

## Why the headline is wrong

### 1. The pgvector latency was *originally* measured against a fresh TCP + Postgres handshake per query

The first cut of the harness opened a new psycopg connection inside `search()`. Across 50 queries on macOS Docker, that's ~40 ms of per-query connection overhead — the kind of methodology hole a vendor blog post quietly skips and an honest one calls out.

The harness now reuses one psycopg connection across every query for the entire `BenchSpec` lifecycle (`setup` opens, `teardown` closes), with the SQL `prepare` deduped to a single round-trip. Re-running the demo against the corrected harness drops pgvector's p95 from ~57 ms into the high-single-digit range — at 5K vectors with default HNSW knobs it's actually faster on p95 than Qdrant on this host, which is the kind of inversion you only see once the connection-establishment noise is out of the picture. The numbers in the table at the top of this post have been regenerated against the current code.

### 2. 100 % recall on Qdrant is *not* a quality flex

HNSW with default `m=16` over 5 000 vectors is essentially exact — the graph is dense enough that the search greedy-descends straight to the top-10. The interesting curves only appear at scale (≥ 100 k) and with `ef_search` sweeps. On a 1M-vector MS-MARCO corpus, you'll see recall@10 in the 0.92–0.98 range and the actual Pareto frontier appears.

### 3. "Index size on disk" is a category mistake — and now there's also a memory column

pgvector reports 4.5 MB for the index. Qdrant reports 0 because the storage layout is different: HNSW lives in memory and only spills via mmap. Comparing them as "disk footprint" is meaningless without normalizing on RSS.

The harness now samples RSS via `psutil` and persists `baseline_rss_bytes`, `index_rss_bytes`, `peak_rss_bytes`, and `adapter_memory_bytes` columns into `summary.parquet`. The peak is a running max across every phase (setup, ingest, build_index, warm-up, measured queries) so transient spikes that earlier landed between checkpoints can't hide; the baseline is captured before any adapter work and subtracted from every later sample, so the reported peak is the adapter-attributable delta, not the constant Python interpreter footprint. The new `memory_recall.png` chart plots that delta on the y-axis as the honest companion to the latency-vs-recall Pareto frontier.

**Profiling methodology.** The harness uses two complementary memory signals: a checkpoint-based `_PeakRssTracker` (five samples per spec, cheap) and a `MemorySampler` background thread (100 ms interval, always on) that populates `adapter_memory_bytes` in the parquet. The background sampler catches sub-checkpoint transients — e.g. an HNSW build that allocates 2× the final index size mid-build and frees most of it before the next checkpoint fires. For call-stack profiling, `--profiler py-spy` wraps the bench loop in `py-spy record` and writes a flame graph SVG to the output directory (requires `pip install 'vdbbench[profile]'`; soft-fails with a warning when py-spy is not on PATH). The flame graph is the tool to reach for when `peak_rss_bytes` is high but it's not obvious which call site is responsible — open `profile.svg` in a browser and look for the widest bars in the ingest or query columns. For I/O-bound investigations, `--profiler iostat` captures continuous CPU + disk utilisation stats via the system `iostat` binary into `iostat.txt`. For a detailed memory breakdown that goes beyond RSS, `--profiler ps_mem` adds a third signal: it polls `ps_mem -p <pid>` on a daemon thread at 5-second intervals and appends timestamped snapshots to `ps_mem.log`, showing the private/shared/swap split that `MemorySampler` cannot see. The three profilers are complementary — `MemorySampler` tracks in-process RSS continuously, `ps_mem` reveals how much memory is genuinely exclusive to the bench process versus shared library pages, and `py-spy` pinpoints the call site responsible for any spike.

### 4. $/M-queries: the number that closes the "which is cheaper?" question

Every DB comparison eventually lands on cost. The harness now computes a `cost_per_million_queries_usd` column in `summary.parquet` using the formula:

```
qps                   = total_queries / total_query_seconds   # measured by the harness
hourly_query_capacity = qps * 3600
cost_per_query        = compute_per_hour_usd / hourly_query_capacity + per_query_usd
cost_per_million      = cost_per_query * 1_000_000
```

The price inputs live in `src/vdbbench/pricing.py` and are sourced from official pricing pages (as of 2026-05). Verified numbers: Neon Scale at $0.222/CU-hour (confirmed from neon.com/pricing) and Chroma Cloud at ~$0.0075/TiB queried (confirmed from trychroma.com/pricing). Qdrant Cloud and LanceDB/S3 are marked `verified=False` — the Qdrant public page hides rates behind a calculator, and the AWS S3 page couldn't be scraped cleanly. The unverified entries are not omitted; they're flagged with a `notes` field pointing to the TODO so a future run can confirm or correct them.

This is a **compute-time estimate against public price-list tiers, not a production-measured bill.** The numbers assume the smallest production-grade tier for each service and single-instance compute. Multi-tenant billing, reserved-instance discounts, and egress charges are excluded. That said, it's the same class of estimate vendors use in their own comparisons — just with the assumptions made explicit rather than buried.

The bottom line from the 5K demo: at ~150 QPS on this host, pgvector on Neon Scale runs ~$0.41/M-queries. The number scales inversely with QPS — a faster host or a tuned `ef_search` that lifts QPS will reduce the $/M figure proportionally. The harness captures this because it derives cost from *measured* query time, not a theoretical rate.

## What the harness already does well

Three things I'm proud of even at 5K rows:

1. **The corpus boundary is iron-clad.** `CorpusBundle` rejects orphan qrels, duplicate (qid, pid) pairs, null ids, and silent dtype drift. The fingerprint is order-invariant and includes metadata, and it's frozen at construction so a bundle used as a dict key won't move buckets if the underlying DataFrame is mutated. That's eight separate codex-review iterations of "wait, what if a loader passes integer ids and the orphan qrel has a 0-grade null pid…" — every one a real bug a sloppier loader would hit at scale.

2. **The MS-MARCO loader is memory-bounded.** A naive loader materializes the 8M-row corpus in Python lists before sub-sampling, which OOMs on a laptop. Mine streams the corpus, always keeps every judged passage, and reservoir-samples unjudged via a deterministic blake2b priority hash. Memory is `O(sample_size)` regardless of corpus size.

3. **`EncodedBundle.save()` doesn't allocate 40 M Python floats.** The naive way to write a 100k×384 matrix to parquet is `pa.array(vecs.tolist(), ...)`. That's ~1 GB of Python objects to garbage-collect right after. The harness uses `pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim)` — straight numpy buffer to Arrow, no Python objects.

## Knob-grid Pareto sweep: finding the recall-vs-latency frontier

The demo numbers above use HNSW defaults — a single point in the recall-vs-latency space. The interesting picture emerges when you sweep the adapter's knobs (`ef_search`, `m`, `metric`) across a grid and ask: *for every achievable recall level, what is the lowest latency any configuration delivers?* That subset is the Pareto frontier.

The harness now ships a `vdbbench sweep` command that runs every Cartesian product of a user-supplied parameter grid against the same encoded dataset and records per-trial recall@k, QPS, and p50/p95/p99 latency. A single command against the `exact` (in-process brute-force) adapter produces a 6-point frontier in under a second:

```bash
make sweep  # metric=cosine,l2 x k_neighbors=4,8,16 — 6 trials total
```

The sweep writes `results/sweep/sweep.parquet` and `results/sweep/sweep_pareto.{png,svg}`. The chart renders dominated points as open markers and the Pareto-optimal subset as filled markers connected by a line — so the frontier reads visually as "beyond this line, you're leaving performance on the table."

**How to interpret the chart.** Each point is one (adapter, params) combination. The x-axis is mean recall@k; the y-axis is p95 latency on a log scale. Points to the upper-right are dominated: another configuration achieves the same recall with lower latency, or higher recall with the same latency. The Pareto-frontier line traces the non-dominated subset — the configurations worth deploying, depending on whether your application is recall-sensitive (accept higher latency for better results) or latency-sensitive (accept some recall degradation for faster responses). The gap between the frontier and any specific operating point is the performance you're leaving on the table by not tuning.

The knob-grid runner is adapter-agnostic: the same `SweepSpec` / `run_sweep` API works against any adapter that follows the `VectorStoreAdapter` protocol — swap `"exact"` for `"pgvector"` and pass `ef_search=32,64,128,256` to sweep HNSW's main search-time knob on a real Postgres instance.

### A methodology lesson: defaults aren't always reasonable

The 100K synthetic-Gaussian run (see `MEASURED-ON.md`) is a worked example of why you
have to sweep before reporting. pgvector's default `ef_search=10` gives recall@10=0.06
at 100K vectors and dim=384. That's not a bug; it's a correct consequence of the data:

**Synthetic 384-dim Gaussian vectors have no cluster structure.** HNSW's greedy
traversal exploits cluster structure to short-circuit the graph — that's what makes
HNSW fast on real retrieval data. Uniform random vectors give HNSW nothing to exploit,
so even at ef\_search=256 (the top end of the sweep), recall only reaches 0.272. On
actual MS-MARCO passages at the same dimension, ef\_search=64 routinely hits recall ≥ 0.92.

The sweep over `ef_search ∈ {32, 64, 128, 256}` with fixed `m=16` (see
`results/100k/sweep-tuned/sweep.parquet`) shows the full recall-vs-latency Pareto curve:

| ef\_search | recall@10 | p95 (ms) |
| ---: | ---: | ---: |
| 32 | 0.050 | 25.1 |
| 64 | 0.098 | 35.7 |
| 128 | 0.172 | 38.8 |
| **256** | **0.272** | **70.1** |

The right operating point is ef\_search=256 for this corpus — not because it gives
great recall, but because it is the best achievable on this data without rebuilding
the index with higher `m` or `ef_construction`. The takeaway: **report both the default
run and the tuned run**. A reader who only sees the tuned number wonders why recall is
capped at 0.28; a reader who only sees the default number (0.06) thinks pgvector is
broken. Both numbers together tell the right story: synthetic Gaussian benchmarks are
adversarial for HNSW, and real workloads will look very different.

## Published HTML report

All of the above — methodology, headline metrics, charts, and per-adapter detail
tables — is also available as a self-contained HTML file that opens in any
browser without an internet connection or markdown renderer:
[`results/demo/report.html`](results/demo/report.html).

The report is generated by `make report` (which calls
`vdbbench report html --from results/demo/ --out results/demo/report.html --charts-dir assets`).
Charts are embedded as base64 `data:` URIs, so the file is truly standalone — no
external file dependencies.

## The right way to read this benchmark

Look at the methodology section in the README. Look at the parquet files in `results/`. Look at what *was* missing — the un-pooled pgvector connection, the un-sampled RSS, the silent partial-availability behavior of `--all`. Each of those gaps was a real story about a tradeoff you can only see once you've built the rig. The round-1 + round-2 fixes shipped this month closed those specific holes; the next set (an `ef_search` sweep, a 1M-vector MS-MARCO Pareto frontier) is the work the repo is built to do, not the headline.

The headline numbers come with a [`MEASURED-ON.md`](MEASURED-ON.md) file documenting the exact hardware, OS, and software versions for each committed run; the machine-readable source of truth lives in `results/<run>/bench_manifest.json`.

## Reproduce in 10 minutes

```bash
git clone https://github.com/BishBish123/vector-db-bench.git
cd vector-db-bench
make install
make up
# `make bench-demo` is the recommended one-shot — it threads
# PGVECTOR_PORT / QDRANT_PORT into the bench DSN/URL automatically,
# so `make up PGVECTOR_PORT=5444` followed by `make bench-demo
# PGVECTOR_PORT=5444` is enough to relocate the whole pipeline off
# the default 5433. The expanded form below uses the env-var defaults;
# swap in $PGVECTOR_PORT / $QDRANT_PORT if you've rebound either.
make bench-demo
# ...or run them by hand:
uv run vdbbench prep   --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
uv run vdbbench bench  --encoded data/encoded-demo --out results/demo \
                       --pgvector-dsn postgresql://bench:bench@localhost:${PGVECTOR_PORT:-5433}/bench \
                       --qdrant-url http://localhost:${QDRANT_PORT:-6333}
uv run vdbbench plot   --summary results/demo/summary.parquet --out assets
```

## 5. bge-small vs nomic: the dim/recall/latency tradeoff at a glance

The harness now supports a direct encoder comparison via `vdbbench compare-encoders`.  The two registered encoders are a useful illustration of the embedding model choice tradeoff:

**bge-small-en-v1.5** (`dim=384`) is a compact model (~33 M parameters) that trades some absolute quality for speed and memory efficiency.  At 384 dimensions, a 1 M-vector index costs ~1.5 GB of float32 storage.  MTEB recall scores sit in the high-40s to low-50s on retrieval benchmarks.  Latency impact is minimal — the embedding dimension is small enough that cosine distance is fast even without hardware acceleration.

**nomic-embed-text-v1.5** (`dim=768`) is a larger model that doubles the embedding space.  The 768-dim representation carries more semantic signal: MTEB retrieval scores typically land 8–12 points above bge-small.  The cost is doubled index memory (~3 GB per 1 M vectors), slightly higher ingest throughput cost (more bytes to write), and marginally higher query latency (longer vectors = more FLOP per dot product).  On the 5 K-vector demo the latency difference is noise; on a 1 M-vector MS-MARCO sweep you will see a measurable gap.

The key tradeoff is rarely "which encoder is better" in isolation.  It is "what recall improvement per GB of index RAM and per millisecond of query latency does the larger model buy you?"  For most applications, bge-small's recall is already good enough and the 2× memory savings are significant at 100 k+ vectors.  The `compare-encoders` command makes that tradeoff visible without running two full bench pipelines by hand.

> **Offline note:** The committed smoke artifact at `results/encoder-compare/encoder_comparison.parquet` uses bge only, because nomic requires a ~500 MB model download and `trust_remote_code=True`.  Run `make compare-encoders ENCODERS=bge,nomic` to produce the full two-encoder comparison locally.

## Containerised reproducibility: one command, no excuses

The 5K-demo numbers are believable only if a stranger can reproduce them on a
different machine and get the same answer within measurement noise.  "Clone and
run `make bench-demo`" is one answer, but it puts Python, uv, and Docker
management on the reviewer — and version drift in any layer silently
invalidates the comparison.

The repo now ships a `Dockerfile` (multi-stage: `builder` stage installs uv
and resolves the lockfile via `uv sync --frozen`; `runtime` stage copies
site-packages and the source tree onto python:3.12-slim) and a
`docker-compose.full.yml` that wires together `pgvector/pgvector:0.8.0-pg17`,
`qdrant/qdrant:v1.17.0`, and the `vdbbench` image with startup-order
guarantees via `depends_on: condition: service_healthy`.  A single command
runs the full pipeline:

```bash
./scripts/reproduce.sh
```

The script builds the image, brings up the stack, runs the bench, and — the
part that actually catches drift — diffs the produced `summary.parquet` against
the committed `results/demo/summary.parquet` baseline.  It exits non-zero if
p95 latency deviates more than 50 % or recall@10 deviates more than 0.05
absolute from the reference.  Tolerances are explicit (configurable via env
vars) rather than silent — when a reviewer gets a green exit they know
*exactly* what "reproduces" means.

The drift check works because parquet preserves schema: both the reference and
the reproduction carry the same column set (`db`, `latency_ms_p95`,
`recall_at_k_mean`, …), so a pandas `groupby("db").iloc[0]` comparison is
stable across runs even when row order differs.  Missing adapters (e.g. a
reviewer without lancedb wheels) are flagged explicitly rather than silently
passed.

A GitHub Actions workflow (`.github/workflows/reproducibility.yml`) runs the
smoke path (`--smoke-only`, exact adapter, no Docker-in-Docker) on every push
to `main` and weekly, so drift is caught automatically before it reaches users.

## The 1M MS-MARCO benchmark: methodology, reproduction, and interpreting the placeholder

The demo run (5 000 synthetic vectors) exercises every code path but is too small to
discriminate the adapters.  The interesting curves — where recall@10 drops below 1.0, where
HNSW beats IVF on latency but loses on throughput, where `ef_search` tuning shifts the Pareto
frontier — only appear at ≥100 k vectors on a real retrieval corpus.

**The 1M run** uses the full MS-MARCO passage corpus (8.8 M passages; the `--limit 1000000`
flag keeps every judged passage and reservoir-samples the rest to exactly 1 M) encoded with
`BAAI/bge-small-en-v1.5` (384-dim).  The pipeline is:

1. `vdbbench prep --dataset msmarco --limit 1000000 --out data/encoded-1m` — streams the HF
   dataset, encodes with bge-small, writes a parquet bundle with ground-truth qrels.
2. `vdbbench bench --all ...` — drives pgvector, qdrant, lancedb, and chroma through the same
   `setup → ingest → build_index → warm-up → search × repeats → teardown` lifecycle.
3. `vdbbench report html` — produces a self-contained HTML report.
4. `vdbbench plot` — regenerates the Pareto frontier and per-adapter bar charts.

The entire pipeline is automated by `scripts/run_1m_bench.sh` (press one button) and
mirrored as a GitHub Actions `workflow_dispatch` workflow at
`.github/workflows/full_bench.yml`.

**How to reproduce the 1M run on appropriate hardware** (Linux, ≥50 GB disk, ≥16 GB RAM):

```bash
./scripts/run_1m_bench.sh          # full pipeline, ~4-8 h
./scripts/run_1m_bench.sh --dry-run  # print every step without executing
make bench-1m-real                 # same, via Make
```

**How to interpret the "not yet measured" placeholder.**  Until a 1M run is published,
`MEASURED-ON.md` carries a `## Full run (1M MS-MARCO) — NOT YET MEASURED` section.  This
is intentional: the repo ships the harness and the automation; the numbers come from the
hardware.  Once a run completes, the operator fills in the `bench_manifest.json` fields into
`MEASURED-ON.md` and commits `results/1m/run-<YYYYMMDD>/summary.parquet` alongside.  The
Pareto frontier charts in `assets/1m/` become the canonical comparison.

Until then, the demo numbers (5 K vectors) are the published reference and are honestly
labelled as a smoke test.  The 1M recipe is reproducible — it just hasn't been run on
publication-grade hardware yet.

## Source

[github.com/BishBish123/vector-db-bench](https://github.com/BishBish123/vector-db-bench) — MIT license, raw data published, contributions welcome.

---

*If you spot a methodology hole, please open an issue. The whole point of publishing the harness is that other people get to find what I missed.*
