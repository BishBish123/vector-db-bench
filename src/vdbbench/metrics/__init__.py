"""Retrieval-metric helpers."""

from vdbbench.metrics.retrieval import (
    RetrievalResult,
    aggregate,
    build_qrel_index,
    recall_at_k,
)

__all__ = [
    "RetrievalResult",
    "aggregate",
    "build_qrel_index",
    "recall_at_k",
]
