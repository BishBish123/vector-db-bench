# Vector-DB-Bench — resume bullets

## Defensible (use these)

- **Built a reproducible vector database benchmarking harness** (pgvector, Qdrant, LanceDB, Chroma, exact brute-force) measuring recall@10, p50/p95/p99 query latency, QPS, ingestion throughput, peak RSS, and cost/M-queries — all persisted as committed parquet artifacts with a `MEASURED-ON.md` documenting exact hardware, adapter versions, and container tags; 578 tests collected.

  *Evidence:* `MEASURED-ON.md` (host metadata + adapter versions for each committed run); `results/100k/run-real/summary.parquet` and `results/demo/summary.parquet` (committed); `pytest --collect-only -q` → "578 tests collected."

- **Ran a 100K-vector benchmark** (100 000 synthetic dim-384 vectors, 100 queries, HNSW defaults, warm profile) on an Intel MacBook Air (8 cores, 8 GiB, macOS 15.7.4, colima Docker): pgvector p95 22.4 ms / recall@10 0.060 / QPS 88.5; Qdrant p95 87.6 ms / recall@10 0.778 / QPS 18.1; exact brute-force p95 221.6 ms / recall@10 1.000 / QPS 6.0 — sourced from `results/100k/run-real/summary.parquet`.

  *Evidence:* `results/100k/run-real/summary.parquet` (parquet columns: `recall_at_k_mean`, `latency_ms_p95`, `qps_estimate`); `MEASURED-ON.md` "100K synthetic run" section with full host metadata.

- **Caught and fixed a methodology bug**: first implementation opened a new psycopg connection per query, adding ~40 ms connection-establishment overhead to every pgvector measurement. Fixed to reuse one connection per `BenchSpec` lifecycle (`setup` opens, `teardown` closes) — numbers in the committed parquet reflect the corrected harness, and the bug is documented in `BLOG.md` under "Why the headline is wrong."

  *Evidence:* `BLOG.md` "The pgvector latency was originally measured against a fresh TCP + Postgres handshake per query" section.

- **Designed a memory-bounded MS-MARCO loader**: streams the 8.8M-row corpus, always retains every judged passage, and reservoir-samples unjudged passages via deterministic blake2b priority hash — memory usage is O(sample_size) regardless of corpus size.

  *Evidence:* `BLOG.md` "The MS-MARCO loader is memory-bounded" section; `src/vdbbench/corpus/beir.py`.

---

## Stretch / claim with caveat (use cautiously)

- **"Cost-per-million-queries column from real pricing"** — `summary.parquet` has `cost_per_million_queries_usd`; `src/vdbbench/pricing.py` sources prices from official pages. Qdrant Cloud and LanceDB/S3 entries are marked `verified=False` in the code and `BLOG.md` notes this. Neon Scale ($0.222/CU-hour) and Chroma Cloud (~$0.0075/TiB) are marked verified.

  *Pushback:* "These numbers are real billing?" — honest answer: "They're compute-time estimates against public price-list tiers — same methodology vendors use, but single-instance, no reserved discounts or egress. Unverified entries are flagged."

- **"Pareto frontier sweep"** — `vdbbench sweep` command exists, `results/sweep/sweep.parquet` is committed, the chart renders. What is not defensible: a multi-adapter Pareto frontier at 100K+ vectors — the committed sweep is against the `exact` adapter only (`make sweep`). A real pgvector + Qdrant sweep requires running the full bench pipeline.

---

## DO NOT claim

- **"1M MS-MARCO benchmark results"** — `MEASURED-ON.md` explicitly states: "Full run (1M MS-MARCO) — NOT YET MEASURED." The pipeline, scripts, and GitHub Actions workflow exist (`scripts/run_1m_bench.sh`, `.github/workflows/full_bench.yml`) but no `results/1m/` parquet is committed.

  *Alternate:* "Built end-to-end 1M MS-MARCO pipeline (`vdbbench prep --dataset msmarco --limit 1000000`); automation script and CI workflow are committed; run requires Linux host with ≥50 GB disk and ≥16 GB RAM."

- **"pgvector recall@10 0.06 is a quality finding"** — the low recall is an artifact of HNSW defaults (ef_search=10, tight for 100K high-dimensional synthetic data), not a meaningful quality comparison. `MEASURED-ON.md` notes this explicitly. Claiming it as "pgvector has poor recall" is misleading.

  *Alternate:* "pgvector HNSW-default recall@10 is 0.06 at ef_search=10 on 100K synthetic dim-384 vectors; Qdrant default ef_search is 128 (more generous), yielding 0.778 — the delta is a parameter choice, not an inherent quality gap."

- **"4-adapter comparison at 100K"** — the committed 100K run has pgvector, Qdrant, and exact brute-force. LanceDB and Chroma are in the adapter registry (`src/vdbbench/adapters/`) but their wheels are unavailable on Intel macOS; the 100K parquet has 3 rows, not 4 or 5.

  *Alternate:* "3-adapter benchmark at 100K vectors (pgvector, Qdrant, exact); LanceDB and Chroma adapters implemented but not benchmarked on this host (no Intel macOS wheels)."

- **"bge-small encoder comparison numbers"** — `results/encoder-compare/encoder_comparison.parquet` uses bge only; the nomic comparison requires `trust_remote_code=True` and a ~500 MB download, not committed. The BLOG says "Offline note: The committed smoke artifact uses bge only."

---

## How to defend each bullet in an interview

**Bullet 1 — reproducible harness, committed parquet, 578 tests:**
> "The key design choice is that results are parquet, not screenshots. `results/100k/run-real/summary.parquet` is committed — you can run `pd.read_parquet('results/100k/run-real/summary.parquet')` and see the same numbers I'm quoting. `MEASURED-ON.md` documents the host (macOS-15.7.4-x86_64, 8 cores, 8 GiB, Python 3.12.13, pgvector 0.4.2, qdrant-client 1.17.1). The 578 tests cover the harness internals — `CorpusBundle` validation, encoder registry, sweep runner, report generation."

**Bullet 2 — 100K numbers, all sourced from parquet:**
> "The parquet columns are: `recall_at_k_mean`, `latency_ms_p50/p95/p99`, `qps_estimate`, `ingest_throughput_vps`, `peak_rss_bytes`. I can read them live: pgvector `latency_ms_p95` = 22.40, `recall_at_k_mean` = 0.060; Qdrant `latency_ms_p95` = 87.58, `recall_at_k_mean` = 0.778. The pgvector recall is low because HNSW ef_search defaults to 10 — very tight for 100K dim-384 random vectors. Qdrant defaults to ef_search=128, which is why its recall is 13x higher at 5x the latency."

**Bullet 3 — methodology bug caught and fixed:**
> "The first version of `search()` in the pgvector adapter opened a new `asyncpg.connect()` on every call. On macOS Docker that's ~40 ms of TCP + Postgres auth per query. The fix is in `src/vdbbench/adapters/pgvector.py`: `setup()` opens the connection, `search()` reuses it, `teardown()` closes it. BLOG.md documents this explicitly and notes that the pre-fix p95 was ~57 ms vs. the post-fix 22 ms. The committed parquet reflects the corrected harness."

**Bullet 4 — memory-bounded MS-MARCO loader:**
> "The naive approach: `list(corpus_iter())` materializes 8.8M passages in memory before subsampling — OOM on a laptop. My loader in `src/vdbbench/corpus/beir.py` streams via the HuggingFace datasets API, always keeps judged passages (the ones with qrels), and reservoir-samples unjudged via `blake2b(pid) % 1` thresholded at `sample_size / corpus_size`. Memory is O(sample_size) at any point. Also: `EncodedBundle.save()` uses `pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim)` instead of `pa.array(vecs.tolist(), ...)` — avoids materializing 40M Python floats."
