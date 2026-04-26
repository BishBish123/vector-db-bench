"""End-to-end synthetic smoke run for CI.

Exercises corpus → encode → save/load → bench → plot offline (no
Postgres, no Qdrant, no model download). The encode step goes through
the deterministic in-package `FakeEncoder` so the script genuinely
covers `encode_corpus()` and the on-disk round-trip in addition to the
bench + plot wiring. Uses the reference brute-force in-memory adapter
so the script runs on every platform in a few seconds.

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
from vdbbench.embed.encoder import FakeEncoder, encode_corpus, load_encoded_bundle
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

    # 1) Generate the bundle. We only use `synth.bundle` — the synthetic
    #    vectors are discarded because the smoke run encodes via
    #    `FakeEncoder` to exercise the real encoding path. (Consequence:
    #    smoke recall numbers are not meaningful — this script verifies
    #    that the pipeline *runs*, not that it scores well.)
    synth = generate_synthetic(
        SyntheticConfig(
            n_passages=args.n_passages,
            n_queries=args.n_queries,
            dim=args.dim,
        )
    )
    bundle = synth.bundle

    # 2) Encode the bundle through the production code path.
    encoder = FakeEncoder(dim=min(args.dim, 64))
    encoded = encode_corpus(bundle, encoder, batch_size=64)

    # 3) Save + reload to confirm the on-disk round-trip is intact. Bench
    #    runs against the reloaded bundle so the smoke covers the full
    #    "what users actually run" sequence, not just the in-memory path.
    encoded_root = args.out / "encoded"
    encoded.save(encoded_root)
    reloaded = load_encoded_bundle(encoded_root)
    if reloaded.bundle.fingerprint() != encoded.bundle.fingerprint():
        raise SystemExit("smoke: encoded round-trip changed bundle fingerprint")
    if not np.array_equal(reloaded.passage_vectors, encoded.passage_vectors):
        raise SystemExit("smoke: encoded round-trip changed passage vectors")

    # 4) Bench + plot.
    spec = BenchSpec(adapter=_MemAdapter(), k=5, profile="warm")
    result = run_bench(reloaded, [spec])
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
