"""Adapter contract test — every available adapter on a minimal corpus.

This is the safety net that catches "this adapter regressed" before the
full bench harness ever runs. It uses an in-memory adapter so it works on
every platform; the matrix of *real* adapters lives behind the
`integration` mark in the per-adapter test files.

The contract is:

    setup(dim, params) -> ingest(ids, vectors) -> build_index() ->
        search(query, k) returns the gold-standard ids ->
        teardown() leaves no residue.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from vdbbench.adapters.base import (
    IndexStats,
    IngestStats,
    VectorStoreAdapter,
)


class _ExactNNAdapter:
    """A reference adapter — brute-force exact NN.

    Stands in for the real adapters in this contract test so the suite
    runs on every platform. Real adapters are tested behind
    `pytest.mark.integration` and need Docker.
    """

    name = "exact-nn-reference"

    def __init__(self) -> None:
        self._dim: int | None = None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._setup_calls = 0
        self._teardown_calls = 0

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._ids = []
        self._mat = None
        self._setup_calls += 1

    def teardown(self) -> None:
        self._ids = []
        self._mat = None
        self._dim = None
        self._teardown_calls += 1

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        if self._dim is None:
            raise RuntimeError("ingest before setup")
        if vectors.shape[1] != self._dim:
            raise ValueError("vector dim mismatch")
        start = time.perf_counter()
        self._ids = list(ids)
        self._mat = vectors.astype(np.float32, copy=False)
        return IngestStats(n_vectors=len(ids), elapsed_s=time.perf_counter() - start)

    def build_index(self) -> IndexStats:
        return IndexStats(elapsed_s=0.0, bytes_disk=0)

    def search(self, query: np.ndarray, k: int) -> list[str]:
        if self._mat is None:
            raise RuntimeError("search before ingest")
        sims = self._mat @ query
        top = np.argsort(-sims)[:k]
        return [self._ids[i] for i in top]

    def memory_footprint_bytes(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Minimal corpus + ground truth
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def _toy_corpus(n: int = 64, dim: int = 16) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Generate `n` passages and 5 queries with brute-force gold pids."""
    rng = np.random.default_rng(0)
    passage_vecs = _l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    query_vecs = _l2_normalize(rng.standard_normal((5, dim)).astype(np.float32))
    pids = [f"p{i:03d}" for i in range(n)]
    sims = query_vecs @ passage_vecs.T
    gold = [pids[int(np.argmax(sims[i]))] for i in range(5)]
    return pids, passage_vecs, query_vecs, gold


# ---------------------------------------------------------------------------
# The contract — applied to every adapter the test discovers.
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> VectorStoreAdapter:
    return _ExactNNAdapter()


class TestAdapterContract:
    """Lifecycle invariants every adapter must satisfy."""

    RECALL_FLOOR = 1.0  # exact NN — for HNSW we'd lower this; ANN regressions
    # below ~0.8 on a 64-vector corpus would still be loud.

    def test_lifecycle_returns_correct_top1(self, adapter: VectorStoreAdapter) -> None:
        pids, passage_vecs, query_vecs, gold = _toy_corpus()
        adapter.setup(dim=passage_vecs.shape[1], params={})
        try:
            ingest = adapter.ingest(pids, passage_vecs)
            assert ingest.n_vectors == len(pids)
            adapter.build_index()
            hits = 0
            for i, expected in enumerate(gold):
                got = adapter.search(query_vecs[i], k=1)
                hits += int(got and got[0] == expected)
            recall = hits / len(gold)
            assert recall >= self.RECALL_FLOOR
        finally:
            adapter.teardown()

    def test_setup_is_idempotent(self, adapter: VectorStoreAdapter) -> None:
        """Calling setup twice must not leave residue from the first call."""
        pids, passage_vecs, _, _ = _toy_corpus(n=8)
        adapter.setup(dim=passage_vecs.shape[1], params={})
        adapter.ingest(pids, passage_vecs)
        adapter.setup(dim=passage_vecs.shape[1], params={})  # reset
        adapter.build_index()
        # No data ingested in this lifecycle; search must return empty.
        # (We allow it to either return [] or raise — both are honest;
        # adapters that returned stale ids would be the failure mode.)
        try:
            result = adapter.search(passage_vecs[0], k=1)
            assert result == []
        except RuntimeError:
            pass
        finally:
            adapter.teardown()

    def test_search_before_setup_raises(self, adapter: VectorStoreAdapter) -> None:
        with pytest.raises((RuntimeError, AttributeError)):
            adapter.search(np.zeros(4, dtype=np.float32), k=1)

    def test_ingest_dim_mismatch_raises(self, adapter: VectorStoreAdapter) -> None:
        adapter.setup(dim=16, params={})
        try:
            with pytest.raises(ValueError, match="dim"):
                adapter.ingest(["p0"], np.zeros((1, 8), dtype=np.float32))
        finally:
            adapter.teardown()

    def test_setup_negative_dim_rejected(self, adapter: VectorStoreAdapter) -> None:
        with pytest.raises(ValueError, match="dim"):
            adapter.setup(dim=0, params={})

    def test_teardown_safe_to_call_repeatedly(self, adapter: VectorStoreAdapter) -> None:
        """Two `teardown` in a row must not raise."""
        adapter.setup(dim=4, params={})
        adapter.teardown()
        adapter.teardown()  # second call must not raise
