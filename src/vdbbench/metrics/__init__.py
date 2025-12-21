"""Retrieval metrics: recall@k, NDCG, MRR, hit-rate + aggregation helpers."""

from vdbbench.metrics.retrieval import (
    QrelIndex,
    RetrievalResult,
    aggregate,
    build_qrel_index,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "QrelIndex",
    "RetrievalResult",
    "aggregate",
    "build_qrel_index",
    "hit_rate",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
