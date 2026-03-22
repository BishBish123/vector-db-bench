"""End-to-end synthetic smoke run for CI.

Exercises corpus → encode → bench → plot offline (no Postgres, no Qdrant,
no model download). Uses the reference brute-force in-memory adapter so
the script runs on every platform in a few seconds.

Invoke as:

    uv run python scripts/smoke_pipeline.py [--out smoke-out]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vdbbench.adapters.base import IndexStats, IngestStats
from vdbbench.bench.runner import BenchSpec, run_bench
from vdbbench.corpus.synthetic import SyntheticConfig, generate_synthetic
from vdbbench.embed.encoder import EncodedBundle
from vdbbench.plot.charts import plot_all


class _MemAdapter:
    """Brute-force exact-NN adapter — same shape as the production ones."""

    name = "mem"

    def __init__(self) -> None:
        self._dim: int | None = None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def setup(self, dim: int, params: dict[str, object]) -> None:
        self._dim = dim
        self._ids = []
        self._mat = None

    def teardown(self) -> None:
        self._ids = []
        self._mat = None
        self._dim = None

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        self._ids = list(ids)
        self._mat = vectors.astype(np.float32, copy=False)
        return IngestStats(n_vectors=len(ids), elapsed_s=0.0)

    def build_index(self) -> IndexStats:
        return IndexStats(elapsed_s=0.0, bytes_disk=0)

    def search(self, query: np.ndarray, k: int) -> list[str]:
        assert self._mat is not None
        sims = self._mat @ query
        return [self._ids[i] for i in np.argsort(-sims)[:k]]

    def memory_footprint_bytes(self) -> int:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("smoke-out"))
    parser.add_argument("--n-passages", type=int, default=128)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    synth = generate_synthetic(
        SyntheticConfig(
            n_passages=args.n_passages,
            n_queries=args.n_queries,
            dim=args.dim,
        )
    )
    encoded = EncodedBundle(
        bundle=synth.bundle,
        passage_vectors=synth.passage_vectors,
        query_vectors=synth.query_vectors,
        encoder_name="synthetic",
    )
    spec = BenchSpec(adapter=_MemAdapter(), k=5, profile="warm")
    result = run_bench(encoded, [spec])
    result.save(args.out)
    plot_all(args.out / "summary.parquet", args.out / "charts")

    summary_path = args.out / "summary.parquet"
    pareto_path = args.out / "charts" / "pareto.png"
    if not summary_path.exists():
        raise SystemExit(f"smoke: missing {summary_path}")
    if not pareto_path.exists():
        raise SystemExit(f"smoke: missing {pareto_path}")
    print(f"smoke pipeline ok — wrote {summary_path} and {pareto_path}")


if __name__ == "__main__":
    main()
