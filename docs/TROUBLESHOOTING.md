# Troubleshooting

Real ops gotchas surfaced during development. Each entry: what you'll see, what's actually happening, how to fix it.

## Docker shared memory (pgvector HNSW)

### Symptom: pgvector container crashes mid-index with `DiskFull` or `no space left on device` on `/dev/shm` during a 100K+ vector run
**Cause:** Docker's default shared-memory allocation is 64 MB. HNSW index builds for 100K × 384-dim vectors require significantly more `/dev/shm`. The default `docker run` / `docker compose up` does not raise this limit automatically.
**Fix:** Pass `--shm-size=1g` to Docker. In `docker-compose.yml` add `shm_size: '1g'` under the pgvector service. The 100K real bench used `shm_size=1g` successfully.
**Source:** `MEASURED-ON.md` 100K run notes; loop_history.md V5 ("REAL ops finding: Docker default shm-size 64MB is too small for pgvector HNSW on 100K×384-dim vectors")

## Port conflicts

### Symptom: `make up` fails with `port is already allocated` for 5433 or 6333/6334
**Cause:** Default ports: pgvector on `5433`, qdrant REST on `6333`, qdrant gRPC on `6334`. These conflict with other dev containers or pgvector instances.
**Fix:** Override all three: `make up PGVECTOR_PORT=5444 QDRANT_PORT=6344 QDRANT_GRPC_PORT=6345`. The same vars flow into `make bench-demo` / `make bench-100k` — pass them consistently.
**Source:** README Prerequisites; `MEASURED-ON.md` 100K run ("Ports: pgvector on 5444, qdrant on 6344/6345")

### Symptom: `make bench-demo` connects but fails with `Connection refused` on pgvector
**Cause:** Port override applied to `make up` but not to `make bench-demo`; the bench CLI defaults to port 5433.
**Fix:** Pass the same override: `make bench-demo PGVECTOR_PORT=5444 QDRANT_PORT=6344`. The Makefile threads these into the DSN/URL automatically; manual CLI invocations require `--pgvector-dsn` with the overridden port explicitly.
**Source:** README Reproduce section port override note

## Low recall on pgvector

### Symptom: `pgvector` recall@10 = 0.06 in the 100K benchmark — looks like a bug
**Cause:** This is a methodology issue, not a code bug. HNSW default `ef_search=10` is very tight relative to k=10 on high-dimensional random data. Qdrant configures `ef_search` more generously by default (ef=128 at index time), hence its higher recall at higher latency.
**Fix:** Sweep `ef_search` to find the recall-vs-latency tradeoff: `make sweep` (exact adapter smoke), or pass custom params via `vdbbench sweep --adapter pgvector --grid ef_search=10,40,100,200`. The `results/sweep/sweep_pareto.{png,svg}` chart shows non-dominated (Pareto-optimal) points.
**Source:** `MEASURED-ON.md` 100K run notes; README Methodology section

## Platform / wheel availability

### Symptom: `make install` silently skips LanceDB and Chroma on Intel macOS
**Cause:** `lancedb` and `chromadb` (via `onnxruntime`) have no macOS x86_64 wheels. `make install` calls `uv sync` with the platform-appropriate extras; on Intel Mac only core + dev extras install. No error is raised.
**Fix:** This is expected. Run `make bench-demo` on Intel Mac — you get a 2-adapter parquet (pgvector + qdrant). For all four adapters run on Linux, Apple Silicon, or WSL2.
**Source:** README Platform support table

### Symptom: `uv run vdbbench prep --encoder nomic` fails with a model loading error about `trust_remote_code`
**Cause:** `nomic-embed-text-v1.5` ships custom modeling code that requires `trust_remote_code=True`. Using `--embed-model nomic-ai/nomic-embed-text-v1.5` directly does NOT set this flag.
**Fix:** Use `--encoder nomic` (the short alias) — it routes through `build_encoder("nomic")` which sets `trust_remote_code=True` automatically. Never use the raw HuggingFace ID for nomic.
**Source:** README Encoders section note; loop_history.md Round 8 ("`nomic trust_remote_code=True` never propagates from registry to CLI" — fixed in R8)

## `--encoder` flag availability

### Symptom: `uv run vdbbench prep --encoder bge` returns `Error: No such option: --encoder`
**Cause:** The `--encoder` flag was added to the `prep` Typer command in Round 8. Older checkouts have only `--embed-model` (raw HuggingFace ID, no registry dispatch).
**Fix:** `git pull` to get the current code; or use `--embed-model BAAI/bge-small-en-v1.5` on the older checkout (nomic will not work with this path — see above).
**Source:** loop_history.md Round 7 ("`--encoder` documented but not wired to `prep`") + Round 8 fix

## Prometheus exporter port leak

### Symptom: Second `vdbbench bench --prometheus-port 9100` call in the same process raises `OSError: [Errno 98] Address already in use`
**Cause:** Older code started the Prometheus HTTP server but never shut it down, so the port stays bound across multiple `run_bench` calls in one process.
**Fix:** Pull latest — `start_exporter` now returns a `WSGIServer` handle; the bench runner wraps it in `try/finally` and calls `stop_exporter()` to release the port. If you're calling `start_exporter` directly, use the returned handle and call `stop_exporter(handle)` when done.
**Source:** loop_history.md R5 codex ("prom start_exporter has no shutdown handle") + R5 codex follow-up fix
