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

## Full run

Not yet measured. When the 1 M-vector MS-MARCO sweep is published,
this section will quote the same fields out of
`results/full/bench_manifest.json`.
