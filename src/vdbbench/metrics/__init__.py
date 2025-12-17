"""Retrieval-metric helpers."""

from vdbbench.metrics.retrieval import (
    RetrievalResult,
    aggregate,
    build_qrel_index,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "RetrievalResult",
    "aggregate",
    "build_qrel_index",
    "hit_rate",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
