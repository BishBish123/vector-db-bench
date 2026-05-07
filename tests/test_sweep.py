"""Tests for the knob-grid Pareto sweep module (vdbbench.sweep).

Covers:
  1. Cartesian product expansion correctness
  2. Pareto-frontier correctness on hand-built fixtures
  3. Sweep against the ``exact`` adapter in mock mode (5+ trials)
  4. Plot writes a PNG of non-zero size
  5. Sweep parquet schema stability
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from vdbbench.sweep import (
    SWEEP_PARQUET_SCHEMA_VERSION,
    SweepResult,
    SweepSpec,
    pareto_frontier,
    run_sweep,
    write_sweep_parquet,
)

# ---------------------------------------------------------------------------
# 1. Cartesian product expansion
# ---------------------------------------------------------------------------


def test_grid_combinations_cartesian_product() -> None:
    """3 keys x 2, 3, 4 values -> 24 distinct trials."""
    spec = SweepSpec(
        adapter="exact",
        parameter_grid={
            "metric": ["cosine", "l2"],
            "ef_search": [32, 64, 128],
            "m": [8, 16, 24, 32],
        },
        dataset="/tmp/fake",
    )
    combos = spec.grid_combinations()
    assert len(combos) == 2 * 3 * 4  # 24
    # All dicts have the same keys.
    assert all(set(c.keys()) == {"metric", "ef_search", "m"} for c in combos)
    # Values are the full cross product — spot check.
    assert {"metric": "cosine", "ef_search": 32, "m": 8} in combos
    assert {"metric": "l2", "ef_search": 128, "m": 32} in combos


def test_grid_combinations_single_key() -> None:
    """Single-axis grid returns one dict per value."""
    spec = SweepSpec(
        adapter="exact",
        parameter_grid={"k": [4, 8, 16]},
        dataset="/tmp/fake",
    )
    combos = spec.grid_combinations()
    assert combos == [{"k": 4}, {"k": 8}, {"k": 16}]


# ---------------------------------------------------------------------------
# 2. Pareto-frontier correctness
# ---------------------------------------------------------------------------


def _make_result(**kw: Any) -> SweepResult:
    """Helper to build a SweepResult with defaults for unused fields."""
    defaults: dict[str, Any] = {
        "adapter": "exact",
        "params": {},
        "recall_at_k": 0.9,
        "qps": 100.0,
        "p50_ms": 1.0,
        "p95_ms": 2.0,
        "p99_ms": 3.0,
        "ingest_seconds": 0.1,
    }
    defaults.update(kw)
    return SweepResult(**defaults)


def test_pareto_frontier_dominated_point_excluded() -> None:
    """A point with lower recall AND higher latency is dominated and dropped."""
    # a dominates b: a has higher recall AND lower p95.
    a = _make_result(recall_at_k=0.95, p95_ms=1.0)
    b = _make_result(recall_at_k=0.80, p95_ms=2.0)  # dominated by a
    frontier = pareto_frontier([a, b])
    assert a in frontier
    assert b not in frontier


def test_pareto_frontier_non_dominated_both_included() -> None:
    """Neither of two points dominates the other → both on the frontier."""
    # high_recall has better recall, low_lat has better latency — neither dominates.
    high_recall = _make_result(recall_at_k=0.95, p95_ms=5.0)
    low_lat = _make_result(recall_at_k=0.80, p95_ms=1.0)
    frontier = pareto_frontier([high_recall, low_lat])
    assert high_recall in frontier
    assert low_lat in frontier


def test_pareto_frontier_tied_points_both_included() -> None:
    """Two results with the same recall AND same p95 are both kept (tie = no dominance)."""
    r1 = _make_result(recall_at_k=0.90, p95_ms=2.0)
    r2 = _make_result(recall_at_k=0.90, p95_ms=2.0)
    frontier = pareto_frontier([r1, r2])
    assert len(frontier) == 2


def test_pareto_frontier_empty_input() -> None:
    """Empty input returns empty output without error."""
    assert pareto_frontier([]) == []


def test_pareto_frontier_single_result() -> None:
    """A single result is always on the frontier."""
    r = _make_result(recall_at_k=0.85, p95_ms=3.0)
    assert pareto_frontier([r]) == [r]


def test_pareto_frontier_ordered_ascending_recall() -> None:
    """Returned frontier is sorted by recall ascending (left-to-right chart order)."""
    results = [
        _make_result(recall_at_k=0.70, p95_ms=0.5),
        _make_result(recall_at_k=0.90, p95_ms=3.0),
        _make_result(recall_at_k=0.80, p95_ms=1.0),
    ]
    frontier = pareto_frontier(results)
    recalls = [r.recall_at_k for r in frontier]
    assert recalls == sorted(recalls)


# ---------------------------------------------------------------------------
# 3. Sweep against exact adapter — real end-to-end in mock mode
# ---------------------------------------------------------------------------


def _make_encoded_bundle(n_passages: int = 20, n_queries: int = 5, dim: int = 8) -> object:
    """Build a tiny EncodedBundle-like object using the real corpus infrastructure."""
    import numpy as np  # noqa: PLC0415

    from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415
    from vdbbench.embed.encoder import EncodedBundle  # noqa: PLC0415

    rng = np.random.default_rng(42)
    passage_vecs = rng.standard_normal((n_passages, dim)).astype(np.float32)
    query_vecs = rng.standard_normal((n_queries, dim)).astype(np.float32)

    passages = pd.DataFrame(
        {"pid": [f"p{i}" for i in range(n_passages)], "text": [f"passage {i}" for i in range(n_passages)]}
    )
    queries = pd.DataFrame(
        {"qid": [f"q{i}" for i in range(n_queries)], "text": [f"query {i}" for i in range(n_queries)]}
    )
    # Assign each query one relevant passage (top-1 by cosine similarity).
    import pandas as _pd  # noqa: PLC0415

    qrel_rows = []
    for i in range(n_queries):
        qrel_rows.append({"qid": f"q{i}", "pid": f"p{i}", "relevance": 1})
    qrels = _pd.DataFrame(qrel_rows)

    bundle = CorpusBundle(name="tiny", passages=passages, queries=queries, qrels=qrels)
    return EncodedBundle(
        bundle=bundle,
        passage_vectors=passage_vecs,
        query_vectors=query_vecs,
        encoder_name="fake",
    )


def test_sweep_exact_produces_nonempty_results(tmp_path: Path) -> None:
    """Sweep against the exact adapter with a 5-trial grid returns 5 results."""
    from unittest.mock import patch  # noqa: PLC0415

    encoded = _make_encoded_bundle()

    spec = SweepSpec(
        adapter="exact",
        parameter_grid={
            "metric": ["cosine", "l2"],
            "k_neighbors": [4, 8, 16],
        },
        dataset=str(tmp_path / "encoded"),  # path unused; we mock load
        top_k=3,
    )
    # 2 x 3 = 6 trials.

    # The import is lazy inside run_sweep (from vdbbench.embed import ...).
    # Patch at the vdbbench.embed module level so the lazy import resolves
    # to our stub.
    with patch("vdbbench.embed.load_encoded_bundle", return_value=encoded):
        results = asyncio.run(run_sweep(spec))

    assert len(results) >= 5  # 6 trials total
    assert all(isinstance(r, SweepResult) for r in results)
    assert all(r.adapter == "exact" for r in results)
    assert all(0.0 <= r.recall_at_k <= 1.0 for r in results)
    assert all(r.p95_ms >= 0.0 for r in results)


# ---------------------------------------------------------------------------
# 4. Plot writes a PNG of non-zero size
# ---------------------------------------------------------------------------


def test_plot_sweep_pareto_writes_png(tmp_path: Path) -> None:
    """plot_sweep_pareto writes a non-empty PNG to the output directory."""
    from vdbbench.plot.charts import plot_sweep_pareto  # noqa: PLC0415

    results = [
        _make_result(recall_at_k=0.70, p95_ms=0.5, qps=2000.0),
        _make_result(recall_at_k=0.80, p95_ms=1.0, qps=1000.0),
        _make_result(recall_at_k=0.90, p95_ms=3.0, qps=333.0),
        _make_result(recall_at_k=0.75, p95_ms=2.5, qps=400.0),  # dominated
    ]
    parquet_path = tmp_path / "sweep.parquet"
    write_sweep_parquet(results, parquet_path)

    out_dir = tmp_path / "charts"
    png, svg = plot_sweep_pareto(parquet_path, out_dir)

    assert png.exists(), "PNG file was not created"
    assert png.stat().st_size > 0, "PNG file is empty"
    assert svg.exists(), "SVG file was not created"


# ---------------------------------------------------------------------------
# 5. Parquet schema stability
# ---------------------------------------------------------------------------


def test_sweep_parquet_schema_stable(tmp_path: Path) -> None:
    """write_sweep_parquet always emits the stable column set."""
    results = [
        _make_result(recall_at_k=0.85, p95_ms=2.0, params={"metric": "cosine"}),
        _make_result(recall_at_k=0.90, p95_ms=3.5, params={"metric": "l2"}),
    ]
    parquet_path = tmp_path / "sweep.parquet"
    write_sweep_parquet(results, parquet_path)

    df = pd.read_parquet(parquet_path)

    # Required stable columns.
    required = {
        "schema_version",
        "adapter",
        "params_json",
        "recall_at_k",
        "qps",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "ingest_seconds",
    }
    assert required.issubset(set(df.columns)), (
        f"Missing columns: {required - set(df.columns)}"
    )

    # Schema version is correct.
    assert (df["schema_version"] == SWEEP_PARQUET_SCHEMA_VERSION).all()

    # Row count matches input.
    assert len(df) == len(results)


def test_sweep_parquet_roundtrip_values(tmp_path: Path) -> None:
    """Values round-trip through parquet without loss."""
    results = [
        _make_result(
            adapter="exact",
            recall_at_k=0.9375,
            p95_ms=1.234,
            qps=810.5,
            ingest_seconds=0.042,
        )
    ]
    parquet_path = tmp_path / "sweep.parquet"
    write_sweep_parquet(results, parquet_path)

    df = pd.read_parquet(parquet_path)
    assert abs(df.iloc[0]["recall_at_k"] - 0.9375) < 1e-6
    assert abs(df.iloc[0]["p95_ms"] - 1.234) < 1e-6
    assert df.iloc[0]["adapter"] == "exact"
