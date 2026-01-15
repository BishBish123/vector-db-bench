"""Integration tests for the LanceDB adapter.

LanceDB is embedded (no service), so the only requirement is that
`lancedb` is importable. The tests skip automatically on platforms where
the wheel isn't published (Intel macOS).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

lancedb = pytest.importorskip(
    "lancedb",
    reason="LanceDB has no wheel for this platform (e.g. Intel macOS); "
    "run on Linux / arm64 macOS / Windows.",
)

from vdbbench.adapters.lancedb import LanceDBAdapter  # noqa: E402

pytestmark = pytest.mark.integration


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[LanceDBAdapter]:
    a = LanceDBAdapter(path=tmp_path / f"db_{uuid.uuid4().hex[:8]}")
    yield a
    with contextlib.suppress(Exception):
        a.teardown()


class TestLanceDBLifecycle:
    def test_setup_idempotent(self, adapter: LanceDBAdapter) -> None:
        adapter.setup(dim=8, params={"index": "none"})
        adapter.setup(dim=8, params={"index": "none"})

    def test_unknown_metric_rejected(self, adapter: LanceDBAdapter) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            adapter.setup(dim=4, params={"metric": "manhattan"})

    def test_unknown_index_kind_rejected(self, adapter: LanceDBAdapter) -> None:
        with pytest.raises(ValueError, match="unknown index kind"):
            adapter.setup(dim=4, params={"index": "hnsw"})  # not supported

    def test_zero_knob_rejected(self, adapter: LanceDBAdapter) -> None:
        with pytest.raises(ValueError, match="num_partitions must be positive"):
            adapter.setup(dim=4, params={"num_partitions": 0})


class TestLanceDBIngestSearch:
    def test_round_trip_no_index(self, adapter: LanceDBAdapter) -> None:
        adapter.setup(dim=16, params={"index": "none"})
        ids = [f"p{i}" for i in range(50)]
        vecs = _random_vectors(50, 16)
        stats = adapter.ingest(ids, vecs, batch_size=10)
        assert stats.n_vectors == 50

        result = adapter.search(vecs[7], k=5)
        # Without an ANN index, LanceDB does a brute-force scan and returns
        # the exact top-k — searching for a known vector ranks it #1.
        assert result[0] == "p7"
        assert len(result) == 5

    def test_dim_mismatch_rejected(self, adapter: LanceDBAdapter) -> None:
        adapter.setup(dim=8, params={"index": "none"})
        with pytest.raises(ValueError, match="dim"):
            adapter.ingest(["p0"], _random_vectors(1, 16))

    def test_search_before_setup_rejected(self, adapter: LanceDBAdapter) -> None:
        with pytest.raises(RuntimeError, match="setup"):
            adapter.search(np.zeros(8, dtype=np.float32), k=5)
