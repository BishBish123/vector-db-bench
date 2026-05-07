"""Custom pgvector ef_search sweep script.

Runs the PgVectorAdapter through a Cartesian product of ef_search values (and
optionally m values) and writes a sweep.parquet + pareto chart to --out.

Usage:
    uv run python scripts/pgvector_sweep.py \
        --dsn postgresql://bench:bench@localhost:5444/bench \
        --encoded data/encoded-100k \
        --out results/100k/sweep-tuned \
        --ef-search 32,64,128,256 \
        --m 16

This is a one-off helper because `vdbbench sweep` only supports the offline
exact/memory adapter today; pgvector requires a live container.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run_pgvector_trial(
    dsn: str,
    encoded_path: str,
    ef_search: int,
    m: int,
    top_k: int = 10,
) -> dict:
    """Run one (ef_search, m) trial and return metrics dict."""
    from vdbbench.adapters.pgvector import PgVectorAdapter
    from vdbbench.embed import load_encoded_bundle
    from vdbbench.metrics.retrieval import aggregate, build_qrel_index, recall_at_k

    encoded = load_encoded_bundle(encoded_path)
    bundle = encoded.bundle
    passage_vectors = encoded.passage_vectors
    query_vectors = encoded.query_vectors

    pids: list[str] = bundle.passages["pid"].astype(str).tolist()
    qids: list[str] = bundle.queries["qid"].astype(str).tolist()
    qrel_index = build_qrel_index(bundle.qrels)
    dim = int(passage_vectors.shape[1])

    params = {
        "index": "hnsw",
        "metric": "cosine",
        "m": m,
        "ef_construction": 64,
        "ef_search": ef_search,
    }

    adapter = PgVectorAdapter(dsn=dsn)
    print(f"  [trial] ef_search={ef_search} m={m}: setting up...", flush=True)
    t0 = time.perf_counter()
    adapter.setup(dim, params)
    adapter.ingest(pids, passage_vectors)
    adapter.build_index()
    ingest_s = time.perf_counter() - t0
    print(f"  [trial] ef_search={ef_search} m={m}: ingest done in {ingest_s:.1f}s, querying...", flush=True)

    # Warm-up pass
    for i in range(min(10, len(qids))):
        adapter.search(query_vectors[i], top_k)

    # Measured pass
    latencies_ms: list[float] = []
    recalls: list[float] = []
    for i, qid in enumerate(qids):
        qvec = query_vectors[i]
        t_q = time.perf_counter()
        retrieved = adapter.search(qvec, top_k)
        latency_ms = (time.perf_counter() - t_q) * 1000.0
        latencies_ms.append(latency_ms)
        if qid in qrel_index:
            rel = qrel_index[qid]
            recalls.append(recall_at_k(retrieved, rel, top_k))

    adapter.teardown()

    arr = np.array(latencies_ms, dtype=np.float64)
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mean_ms = float(np.mean(arr))
    qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    result = {
        "adapter": "pgvector",
        "params_json": json.dumps({"ef_search": ef_search, "m": m}, sort_keys=True),
        "ef_search": ef_search,
        "m": m,
        "recall_at_k": recall_mean,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "ingest_seconds": ingest_s,
        "schema_version": 1,
    }
    print(
        f"  [trial] ef_search={ef_search} m={m}: recall@{top_k}={recall_mean:.3f} "
        f"p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms qps={qps:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="pgvector ef_search sweep")
    parser.add_argument("--dsn", default="postgresql://bench:bench@localhost:5444/bench")
    parser.add_argument("--encoded", default="data/encoded-100k")
    parser.add_argument("--out", default="results/100k/sweep-tuned")
    parser.add_argument("--ef-search", default="32,64,128,256")
    parser.add_argument("--m", default="16")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    ef_search_values = [int(x.strip()) for x in args.ef_search.split(",")]
    m_values = [int(x.strip()) for x in args.m.split(",")]

    print(f"pgvector sweep: ef_search={ef_search_values} m={m_values}", flush=True)
    print(f"  DSN: {args.dsn}", flush=True)
    print(f"  encoded: {args.encoded}", flush=True)
    print(f"  out: {args.out}", flush=True)

    results = []
    for m in m_values:
        for ef in ef_search_values:
            result = run_pgvector_trial(
                dsn=args.dsn,
                encoded_path=args.encoded,
                ef_search=ef,
                m=m,
                top_k=args.top_k,
            )
            results.append(result)

    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    sweep_parquet = out_dir / "sweep.parquet"
    df.to_parquet(sweep_parquet, index=False)
    print(f"\nWrote {len(results)} trials to {sweep_parquet}", flush=True)

    print("\n=== Sweep Results Table ===")
    print(f"{'ef_search':>10} {'m':>4} {'recall@10':>12} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10} {'qps':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['ef_search']:>10} {r['m']:>4} {r['recall_at_k']:>12.4f} "
            f"{r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} "
            f"{r['qps']:>8.1f}"
        )

    # Find best ef_search: highest recall
    best = max(results, key=lambda r: r["recall_at_k"])
    print(f"\nBest: ef_search={best['ef_search']} m={best['m']} recall@10={best['recall_at_k']:.4f}")

    # Also try to generate a pareto chart
    try:
        from vdbbench.plot.charts import plot_sweep_pareto
        png, svg = plot_sweep_pareto(sweep_parquet, out_dir)
        print(f"Pareto chart: {png.name} + {svg.name}")
    except Exception as e:
        print(f"Warning: could not generate pareto chart: {e}")


if __name__ == "__main__":
    main()
