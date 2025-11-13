"""A no-network synthetic corpus with known nearest-neighbour ground truth.

Used by tests, smoke runs, and CI — exercises the entire pipeline without
needing Hugging Face or model downloads. The "passages" and "queries" are
random unit vectors written out as base64-encoded text fields so the same
parquet schema works as for real text corpora; the brute-force top-k over
the matrix is the gold standard.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vdbbench.corpus.bundle import CorpusBundle


@dataclass(frozen=True)
class SyntheticConfig:
    n_passages: int = 1_000
    n_queries: int = 50
    dim: int = 32
    relevant_k: int = 5  # how many nearest passages count as "relevant" per query
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_passages <= 0:
            raise ValueError("n_passages must be positive")
        if self.n_queries <= 0:
            raise ValueError("n_queries must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.relevant_k <= 0 or self.relevant_k > self.n_passages:
            raise ValueError("relevant_k must be in [1, n_passages]")


@dataclass(frozen=True, eq=False)
class SyntheticCorpus:
    """A bundle plus the dense vector matrices used to derive its qrels.

    Equality and hashing are disabled because `np.ndarray.__eq__` returns an
    array (raises in a boolean context) and ndarrays are unhashable. Compare
    via `bundle.fingerprint()` plus `np.array_equal` on the matrices when
    needed.
    """

    bundle: CorpusBundle
    passage_vectors: np.ndarray  # shape (n_passages, dim), L2-normalized
    query_vectors: np.ndarray  # shape (n_queries, dim),  L2-normalized


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row, preserving the input dtype.

    `np.where(norm == 0.0, 1.0, norm)` would otherwise promote to float64;
    we keep float32 so that the qrels (computed from these vectors) match
    the float32 representation that gets serialized into the bundle.
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True).astype(x.dtype, copy=False)
    norm = np.where(norm == 0.0, np.array(1.0, dtype=x.dtype), norm)
    return (x / norm).astype(x.dtype, copy=False)


def _encode(vec: np.ndarray) -> str:
    """Pack a float32 vector into a base64 string so it round-trips through parquet."""
    return base64.b64encode(vec.astype(np.float32, copy=False).tobytes()).decode("ascii")


def generate_synthetic(cfg: SyntheticConfig) -> SyntheticCorpus:
    """Generate a deterministic synthetic corpus + its brute-force top-k qrels."""
    rng = np.random.default_rng(cfg.seed)
    passage_vecs = _l2_normalize(rng.standard_normal((cfg.n_passages, cfg.dim)).astype(np.float32))
    query_vecs = _l2_normalize(rng.standard_normal((cfg.n_queries, cfg.dim)).astype(np.float32))

    passages = pd.DataFrame(
        {
            "pid": [f"p{i:08d}" for i in range(cfg.n_passages)],
            "text": [_encode(v) for v in passage_vecs],
        }
    )
    queries = pd.DataFrame(
        {
            "qid": [f"q{i:08d}" for i in range(cfg.n_queries)],
            "text": [_encode(v) for v in query_vecs],
        }
    )

    # Brute-force cosine similarity for ground truth (vectors already normalized
    # so the inner product is the cosine).
    sims = query_vecs @ passage_vecs.T
    top_idx = np.argpartition(-sims, kth=cfg.relevant_k - 1, axis=1)[:, : cfg.relevant_k]
    # argpartition gives unsorted indices; sort each row by similarity descending
    # so the highest-relevance pid appears first per query — handy for graded recall.
    row_idx = np.arange(cfg.n_queries)[:, None]
    sorted_within = np.argsort(-sims[row_idx, top_idx], axis=1)
    top_idx_sorted = top_idx[row_idx, sorted_within]

    qrel_rows: list[tuple[str, str, int]] = []
    for q_i in range(cfg.n_queries):
        for rank, p_i in enumerate(top_idx_sorted[q_i]):
            # Higher relevance = closer neighbour. Top hit gets relevance == relevant_k.
            relevance = cfg.relevant_k - rank
            qrel_rows.append((queries.loc[q_i, "qid"], passages.loc[int(p_i), "pid"], relevance))

    qrels = pd.DataFrame(qrel_rows, columns=["qid", "pid", "relevance"])

    bundle = CorpusBundle(
        name=f"synthetic-d{cfg.dim}-n{cfg.n_passages}-q{cfg.n_queries}-s{cfg.seed}",
        passages=passages,
        queries=queries,
        qrels=qrels,
        metadata={
            "kind": "synthetic",
            "dim": cfg.dim,
            "relevant_k": cfg.relevant_k,
            "seed": cfg.seed,
        },
    )
    return SyntheticCorpus(
        bundle=bundle,
        passage_vectors=passage_vecs,
        query_vectors=query_vecs,
    )


__all__ = ["SyntheticConfig", "SyntheticCorpus", "generate_synthetic"]
