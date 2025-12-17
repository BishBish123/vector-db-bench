"""Retrieval quality metrics: recall@k.

These are pure functions over `RetrievalResult` (one per query) and a
`QrelIndex` (qid -> pid -> relevance grade). The bench harness collects
results, then runs `aggregate(...)` to produce the per-DB summary stats
(mean, p50, p95, std) that get reported in the headline charts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# qid -> pid -> relevance grade (>= 0; 0 means "judged not relevant").
QrelIndex = dict[str, dict[str, float]]


@dataclass(frozen=True)
class RetrievalResult:
    """One DB's response to one query, ordered by rank (best first)."""

    qid: str
    retrieved_pids: tuple[str, ...]  # tuple so the dataclass stays hashable

    def __post_init__(self) -> None:
        # Detect repeats early — duplicate pids in the response would inflate
        # recall artificially and confuse NDCG ranking. Adapters should
        # de-dupe before returning, but this is a cheap safety net.
        if len(set(self.retrieved_pids)) != len(self.retrieved_pids):
            raise ValueError(
                f"retrieved_pids for qid={self.qid!r} contains duplicates: {self.retrieved_pids!r}"
            )


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_qrel_index(qrels: pd.DataFrame) -> QrelIndex:
    """Group `qrels` (qid, pid, relevance) into the nested dict the metrics
    expect. Drops rows with `relevance <= 0` because those passages are
    *judged not relevant* — they should not count as positives.
    """
    if qrels.empty:
        return {}
    positives = qrels[qrels["relevance"] > 0]
    index: QrelIndex = {}
    for qid, pid, rel in zip(
        positives["qid"].astype(str),
        positives["pid"].astype(str),
        positives["relevance"].astype(float),
        strict=True,
    ):
        index.setdefault(qid, {})[pid] = float(rel)
    return index


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def _positives(relevant: dict[str, float]) -> set[str]:
    """Return only the pids with `relevance > 0`.

    BEIR-style qrels carry judged negatives as `relevance == 0`; counting
    those as hits would silently inflate every retrieval metric. We filter
    here so callers can pass raw graded maps without going through
    `build_qrel_index` first.
    """
    return {pid for pid, rel in relevant.items() if rel > 0}


def recall_at_k(
    retrieved: tuple[str, ...] | list[str], relevant: dict[str, float], k: int
) -> float:
    """`|top-k ∩ positives| / |positives|`.

    Returns 0.0 when there are no positives so the result is comparable
    across queries (bench harness usually filters those out anyway).
    `relevant` may include `relevance == 0` entries — those are *judged
    negatives* and are explicitly excluded from the numerator and
    denominator.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    positives = _positives(relevant)
    if not positives:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for pid in positives if pid in top_k)
    return hits / len(positives)


def hit_rate(retrieved: tuple[str, ...] | list[str], relevant: dict[str, float], k: int) -> float:
    """1.0 if any *positive* pid appears in the top `k`, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    positives = _positives(relevant)
    if not positives:
        return 0.0
    return float(any(pid in positives for pid in retrieved[:k]))


def mrr(
    retrieved: tuple[str, ...] | list[str], relevant: dict[str, float], k: int | None = None
) -> float:
    """Reciprocal rank of the first *positive* pid; 0.0 if none in top `k`.

    `k=None` means "look through the entire response". Most retrieval
    benchmarks report MRR@10 — pass `k=10` for that.
    """
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    positives = _positives(relevant)
    if not positives:
        return 0.0
    cap = len(retrieved) if k is None else min(k, len(retrieved))
    for rank, pid in enumerate(retrieved[:cap], start=1):
        if pid in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: tuple[str, ...] | list[str], relevant: dict[str, float], k: int) -> float:
    """Normalized DCG @ k with the standard `(2^rel - 1) / log2(rank + 1)` gain.

    Implementation note: sums over the top-k retrieved using their graded
    relevance, then divides by the ideal DCG (top-k by descending grade
    over `relevant`). Returns 0.0 when the ideal DCG is 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(retrieved[:k], start=1):
        rel = relevant.get(pid, 0.0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1)

    # Ideal: top-k positives by grade descending.
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(
        (2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_grades, start=1)
    )
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(values: Iterable[float]) -> dict[str, float]:
    """Reduce a per-query metric to mean / p50 / p95 / std / n.

    Ignores empty inputs by returning all-NaN so callers can spot the
    "no queries had ground truth" case at report time.
    """
    arr = np.fromiter(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }
