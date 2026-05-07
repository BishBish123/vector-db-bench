"""Tuned bench: run pgvector at a specific ef_search + qdrant + exact, write summary.parquet.

This script drives the full run_bench pipeline with a custom pgvector BenchSpec that
sets ef_search to the value chosen from the sweep, alongside qdrant and exact.

Usage:
    uv run python scripts/tuned_bench.py \
        --pgvector-dsn postgresql://bench:bench@localhost:5444/bench \
        --qdrant-url http://localhost:6344 \
        --ef-search 128 \
        --encoded data/encoded-100k \
        --out results/100k-tuned/run-real
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuned pgvector + qdrant + exact bench")
    parser.add_argument(
        "--pgvector-dsn", default="postgresql://bench:bench@localhost:5444/bench"
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6344")
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--encoded", default="data/encoded-100k")
    parser.add_argument("--out", default="results/100k-tuned/run-real")
    parser.add_argument("--profile", default="warm")
    args = parser.parse_args()

    from vdbbench.adapters import ExactAdapter, PgVectorAdapter, QdrantAdapter
    from vdbbench.bench import BenchSpec, run_bench
    from vdbbench.embed import load_encoded_bundle

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = load_encoded_bundle(args.encoded)

    specs = [
        BenchSpec(
            adapter=PgVectorAdapter(dsn=args.pgvector_dsn),
            params={
                "index": "hnsw",
                "metric": "cosine",
                "m": 16,
                "ef_construction": 64,
                "ef_search": args.ef_search,
            },
            k=10,
            repeats=-1,
            label=f"pgvector:hnsw-ef{args.ef_search}",
            profile=args.profile,
        ),
        BenchSpec(
            adapter=QdrantAdapter(url=args.qdrant_url, timeout=120),
            params={"metric": "cosine"},
            k=10,
            repeats=-1,
            label="qdrant:hnsw-default",
            profile=args.profile,
        ),
        BenchSpec(
            adapter=ExactAdapter(),
            params={"metric": "cosine"},
            k=10,
            repeats=-1,
            label="exact:bruteforce",
            profile=args.profile,
        ),
    ]

    print(
        f"Running tuned bench: pgvector ef_search={args.ef_search}, qdrant, exact",
        flush=True,
    )
    print(f"  encoded: {args.encoded}", flush=True)
    print(f"  out: {out}", flush=True)

    result = run_bench(enc, specs, progress=True, tolerate_failures=False, out=out)
    out_path = result.save(out)
    print(f"Wrote results to {out_path}", flush=True)

    # Print summary table
    import pandas as pd

    df = pd.read_parquet(out / "summary.parquet")
    print("\n=== Tuned Bench Results ===")
    print(
        df[["label", "recall_at_k_mean", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
            "ingest_throughput_vps", "index_build_s", "cost_per_million_queries_usd"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
