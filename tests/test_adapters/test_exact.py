"""Unit tests for the in-process ExactAdapter (no DB)."""

from __future__ import annotations

import numpy as np
import pytest

from vdbbench.adapters import ExactAdapter


def _toy_vectors() -> tuple[list[str], np.ndarray]:
    ids = ["p0", "p1", "p2", "p3"]
    mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ids, mat


class TestLifecycle:
    def test_search_before_setup_rejected(self) -> None:
        a = ExactAdapter()
        with pytest.raises(RuntimeError, match="search"):
            a.search(np.zeros(3, dtype=np.float32), k=1)

    def test_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            ExactAdapter().setup(dim=3, params={"metric": "hamming"})

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            ExactAdapter().setup(dim=0, params={})

    def test_filter_arg_rejected(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={})
        ids, mat = _toy_vectors()
        a.ingest(ids, mat)
        with pytest.raises(NotImplementedError):
            a.search(np.zeros(3, dtype=np.float32), k=1, filter={"x": 1})


class TestSearchCorrectness:
    """Brute-force NN — recall@k must be 1.0 by construction. These tests
    pin that contract so a downstream regression in the bench (e.g. a
    typo in the recall metric) shows up here, not silently."""

    def test_cosine_top1_is_self(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={"metric": "cosine"})
        ids, mat = _toy_vectors()
        a.ingest(ids, mat)
        # Querying with p0's own vector — top-1 must be p0.
        top = a.search(mat[0], k=1)
        assert top == ["p0"]

    def test_l2_returns_closest_first(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={"metric": "l2"})
        ids, mat = _toy_vectors()
        a.ingest(ids, mat)
        # p2 = [0.9, 0.1, 0] is closer to p0 than to p1; querying with
        # [1, 0, 0] under L2 must return [p0, p2, ...] before p1 / p3.
        top = a.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert top == ["p0", "p2"]

    def test_inner_top1_is_self(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={"metric": "inner"})
        ids, mat = _toy_vectors()
        a.ingest(ids, mat)
        top = a.search(mat[1], k=1)
        assert top == ["p1"]


class TestMemoryFootprint:
    def test_reports_matrix_nbytes(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={})
        ids, mat = _toy_vectors()
        a.ingest(ids, mat)
        # 4 rows * 3 cols * float32 (4 bytes) = 48 bytes.
        assert a.memory_footprint_bytes() == 4 * 3 * 4

    def test_zero_before_ingest(self) -> None:
        a = ExactAdapter()
        a.setup(dim=3, params={})
        assert a.memory_footprint_bytes() == 0
