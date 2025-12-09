"""Retrieval quality metrics: recall@k.

These are pure functions over `RetrievalResult` (one per query) and a
`QrelIndex` (qid -> pid -> relevance grade). The bench harness collects
results, then runs `aggregate(...)` to produce the per-DB summary stats
(mean, p50, p95, std) that get reported in the headline charts.
"""

from __future__ import annotations

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
