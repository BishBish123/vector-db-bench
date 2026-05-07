"""Knob-grid Pareto sweep — sweep adapter parameters and find the recall-vs-latency frontier.

Defines:
  * ``SweepSpec``     — what to sweep (adapter, grid, dataset config)
  * ``SweepResult``   — one trial's metrics
  * ``run_sweep``     — async entry point; runs every Cartesian product of the grid
  * ``write_sweep_parquet`` — stable parquet schema writer
  * ``pareto_frontier``    — filters results to the non-dominated (recall, latency) subset

Usage from the CLI::

    vdbbench sweep --adapter exact --grid k_neighbors=4,8,16 \\
                   --encoded smoke-out --out results/sweep/
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

SWEEP_PARQUET_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class SweepSpec:
    """Everything needed to run a knob-grid sweep for one adapter.

    ``parameter_grid`` maps each knob name to the list of values to try.
    The sweep runs every Cartesian product of these lists against the same
    encoded dataset.

    Example::

        SweepSpec(
            adapter="exact",
            parameter_grid={"metric": ["cosine", "l2"], "k_neighbors": [4, 8, 16]},
            dataset="data/encoded-demo",
            corpus_size=5000,
            n_queries=100,
            top_k=10,
        )
    """

    adapter: str
    parameter_grid: dict[str, list[Any]]
    dataset: str  # path to encoded bundle directory
    corpus_size: int = 5000
    n_queries: int = 100
    top_k: int = 10

    def __post_init__(self) -> None:
        if not self.parameter_grid:
            raise ValueError("parameter_grid must have at least one key")
        if self.corpus_size <= 0:
            raise ValueError("corpus_size must be positive")
        if self.n_queries <= 0:
            raise ValueError("n_queries must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def grid_combinations(self) -> list[dict[str, Any]]:
        """Return a list of dicts — one per Cartesian product of the grid values."""
        keys = list(self.parameter_grid.keys())
        value_lists = [self.parameter_grid[k] for k in keys]
        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*value_lists)]


@dataclass(frozen=True)
class SweepResult:
    """Metrics from a single (adapter, params) trial in the sweep."""

    adapter: str
    params: dict[str, Any]
    recall_at_k: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ingest_seconds: float
    params_json: str = field(default="")

    def __post_init__(self) -> None:
        # Populate params_json from params when not supplied directly.
        if not self.params_json:
            object.__setattr__(
                self, "params_json", json.dumps(self.params, sort_keys=True, default=str)
            )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def pareto_frontier(results: list[SweepResult]) -> list[SweepResult]:
    """Return the subset of *results* that are non-dominated on (recall_at_k, p95_ms).

    A result is dominated when some other result has **both** higher-or-equal
    recall *and* lower-or-equal p95 latency, with at least one inequality
    strict. Only non-dominated (Pareto-optimal) results are returned.

    Ties: if two results share the same recall AND the same p95, **both**
    are included — neither dominates the other.

    Algorithm: sort by recall descending, p95 ascending; walk forward
    keeping a running minimum p95. A row survives if its p95 is strictly
    less than the current best (i.e. it is faster than everything at
    higher-or-equal recall that we've already accepted). On a p95 tie
    at the same recall level, the row where p95 == best_p95 is also kept
    because there's no strict improvement to exclude it.
    """
    if not results:
        return []

    # Sort: higher recall first; among equal recall, lower p95 first.
    sorted_results = sorted(results, key=lambda r: (-r.recall_at_k, r.p95_ms))

    frontier: list[SweepResult] = []
    best_p95 = float("inf")

    for r in sorted_results:
        if r.p95_ms <= best_p95:
            frontier.append(r)
            best_p95 = r.p95_ms

    # Return left-to-right by recall ascending so callers plotting the
    # frontier see a natural "more recall = more latency" slope.
    return sorted(frontier, key=lambda r: r.recall_at_k)


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def write_sweep_parquet(results: list[SweepResult], path: str | Path) -> Path:
    """Write *results* to a parquet file with a stable, versioned schema.

    The ``schema_version`` column lets future readers detect incompatible
    changes (bumped in :data:`SWEEP_PARQUET_SCHEMA_VERSION`).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for r in results:
        # Build a flat row with only stable scalar columns — pyarrow cannot
        # serialise an arbitrary dict (the ``params`` field) as a parquet
        # struct when the schema is empty or inconsistent across rows.
        # The full params payload lives in ``params_json`` (JSON string).
        row: dict[str, object] = {
            "schema_version": SWEEP_PARQUET_SCHEMA_VERSION,
            "adapter": r.adapter,
            "params_json": json.dumps(r.params, sort_keys=True, default=str),
            "recall_at_k": float(r.recall_at_k),
            "qps": float(r.qps),
            "p50_ms": float(r.p50_ms),
            "p95_ms": float(r.p95_ms),
            "p99_ms": float(r.p99_ms),
            "ingest_seconds": float(r.ingest_seconds),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Column order is already stable (dicts preserve insertion order in
    # Python 3.7+, and we build rows in the stable_cols order above).
    df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


async def run_sweep(spec: SweepSpec) -> list[SweepResult]:
    """Run every Cartesian product of *spec.parameter_grid* and return all metrics.

    Each combination is run synchronously inside the async wrapper — the
    async signature is kept so callers can ``await`` this in an event loop
    or compose it with other async work. The bench itself is CPU-bound
    (numpy brute-force or socket round-trips), so running combinations
    sequentially is correct; parallelism would be wrong for shared DB state.

    The function loads the encoded bundle once and reuses it for every
    combination, so startup cost is O(1) per sweep rather than O(N_trials).
    """
    from vdbbench.embed import load_encoded_bundle  # noqa: PLC0415

    encoded = load_encoded_bundle(spec.dataset)

    combinations = spec.grid_combinations()
    results: list[SweepResult] = []

    for params in combinations:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_single_trial, spec, encoded, params
        )
        results.append(result)

    return results


def _run_single_trial(
    spec: SweepSpec,
    encoded: object,
    params: dict[str, Any],
) -> SweepResult:
    """Run one (adapter, params) combination and return its metrics.

    Drives the full ingest → index → query lifecycle against the ExactAdapter
    (and any other adapter that follows the VectorStoreAdapter protocol).
    """
    from vdbbench.adapters import ExactAdapter  # noqa: PLC0415
    from vdbbench.metrics.retrieval import (  # noqa: PLC0415
        aggregate,
        build_qrel_index,
        recall_at_k,
    )

    # Resolve the adapter.
    adapter_name = spec.adapter.lower()
    if adapter_name in ("exact", "memory"):
        adapter = ExactAdapter()
    else:
        raise ValueError(
            f"sweep: adapter {spec.adapter!r} is not supported without Docker; "
            "only 'exact' / 'memory' runs offline. Pass a supported adapter name."
        )

    # Access EncodedBundle attributes via duck typing to avoid a circular import.
    bundle = encoded.bundle  # type: ignore[attr-defined]
    passage_vectors = encoded.passage_vectors  # type: ignore[attr-defined]
    query_vectors = encoded.query_vectors  # type: ignore[attr-defined]

    pids: list[str] = bundle.passages["pid"].astype(str).tolist()
    qids: list[str] = bundle.queries["qid"].astype(str).tolist()
    qrel_index = build_qrel_index(bundle.qrels)

    dim = int(passage_vectors.shape[1])

    # Build effective params: merge sweep params with required adapter params.
    # For ExactAdapter, only 'metric' is relevant; extra keys are silently ignored.
    effective_params: dict[str, object] = {"metric": "cosine"}
    effective_params.update(params)

    # Ingest.
    t0 = time.perf_counter()
    adapter.setup(dim, effective_params)
    adapter.ingest(pids, passage_vectors)
    adapter.build_index()
    ingest_seconds = time.perf_counter() - t0

    # Query.
    latencies_ms: list[float] = []
    recalls: list[float] = []

    for i, qid in enumerate(qids):
        qvec = query_vectors[i]
        t_q = time.perf_counter()
        retrieved = adapter.search(qvec, spec.top_k)
        latency_ms = (time.perf_counter() - t_q) * 1000.0
        latencies_ms.append(latency_ms)
        if qid in qrel_index:
            rel = qrel_index[qid]
            recalls.append(recall_at_k(retrieved, rel, spec.top_k))

    adapter.teardown()

    # Aggregate.
    agg = aggregate(latencies_ms)
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    mean_ms = float(agg["mean"]) if agg["mean"] is not None else 0.0
    qps_val = (1000.0 / mean_ms) if mean_ms > 0 else float("inf")

    arr = np.array(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50)) if len(arr) > 0 else 0.0
    p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0
    p99 = float(np.percentile(arr, 99)) if len(arr) > 0 else 0.0

    return SweepResult(
        adapter=spec.adapter,
        params=dict(params),
        recall_at_k=recall_mean,
        qps=qps_val,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        ingest_seconds=ingest_seconds,
    )
