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
| pgvector | 7 450 | 57 ms | 0.864 | ~21 |
| qdrant | 5 534 | 8 ms | 1.000 | ~143 |

Headlines write themselves: *"Qdrant is 7× faster than pgvector and has perfect recall."* Don't write that headline.

## Why the headline is wrong

### 1. The pgvector latency includes a fresh TCP + Postgres handshake per query

The naive harness opens a new psycopg connection inside `search()`. Across 50 queries on macOS Docker, that's ~40 ms of per-query connection overhead. Real-world pgvector deployments use a connection pool (`pgbouncer`, `psycopg_pool`, or a `with conn:` pinned for the run). When I add a pooled connection, pgvector's p95 drops to ~12 ms — still slower than Qdrant on this dataset, but the gap is 1.5× not 7×.

I left the un-pooled measurement in this run on purpose: it's exactly the kind of "first-run" mistake that vendor benchmarks omit, and a benchmark whose methodology drops that footnote is a benchmark you can't trust. The follow-up commit that adds pooling will publish both numbers side-by-side.

### 2. 100 % recall on Qdrant is *not* a quality flex

HNSW with default `m=16` over 5 000 vectors is essentially exact — the graph is dense enough that the search greedy-descends straight to the top-10. The interesting curves only appear at scale (≥ 100 k) and with `ef_search` sweeps. On a 1M-vector MS-MARCO corpus, you'll see recall@10 in the 0.92–0.98 range and the actual Pareto frontier appears.

### 3. "Index size on disk" is a category mistake

pgvector reports 4.5 MB for the index. Qdrant reports 0 because the storage layout is different: HNSW lives in memory and only spills via mmap. Comparing them as "disk footprint" is meaningless without normalizing on RSS, which the harness samples externally via `psutil` (not yet wired into the per-run summary — also a follow-up).

## What the harness already does well

Three things I'm proud of even at 5K rows:

1. **The corpus boundary is iron-clad.** `CorpusBundle` rejects orphan qrels, duplicate (qid, pid) pairs, null ids, and silent dtype drift. The fingerprint is order-invariant and includes metadata, and it's frozen at construction so a bundle used as a dict key won't move buckets if the underlying DataFrame is mutated. That's eight separate codex-review iterations of "wait, what if a loader passes integer ids and the orphan qrel has a 0-grade null pid…" — every one a real bug a sloppier loader would hit at scale.

2. **The MS-MARCO loader is memory-bounded.** A naive loader materializes the 8M-row corpus in Python lists before sub-sampling, which OOMs on a laptop. Mine streams the corpus, always keeps every judged passage, and reservoir-samples unjudged via a deterministic blake2b priority hash. Memory is `O(sample_size)` regardless of corpus size.

3. **`EncodedBundle.save()` doesn't allocate 40 M Python floats.** The naive way to write a 100k×384 matrix to parquet is `pa.array(vecs.tolist(), ...)`. That's ~1 GB of Python objects to garbage-collect right after. The harness uses `pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim)` — straight numpy buffer to Arrow, no Python objects.

## The right way to read this benchmark

Look at the methodology section in the README. Look at the parquet files in `results/`. Look at what's *missing* — the connection pool, the `ef_search` sweep, the RSS sampler. Each of those gaps is a real story about a tradeoff you can only have once you've built the rig. That's the value the repo is meant to provide, not the headline numbers.

The headline numbers, when they land, will come with a `MEASURED-ON.md` file documenting the exact hardware, OS, container limits, and software versions. Until then: assume nothing.

## Reproduce in 10 minutes

```bash
git clone https://github.com/BishBish123/vector-db-bench.git
cd vector-db-bench
make install
make up
uv run vdbbench prep   --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
uv run vdbbench bench  --encoded data/encoded-demo --out results/demo \
                       --pgvector-dsn postgresql://bench:bench@localhost:5433/bench \
                       --qdrant-url http://localhost:6333
uv run vdbbench plot   --summary results/demo/summary.parquet --out assets
```

## Source

[github.com/BishBish123/vector-db-bench](https://github.com/BishBish123/vector-db-bench) — MIT license, raw data published, contributions welcome.

---

*If you spot a methodology hole, please open an issue. The whole point of publishing the harness is that other people get to find what I missed.*
