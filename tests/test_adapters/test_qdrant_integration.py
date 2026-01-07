"""Integration tests for the Qdrant adapter.

Requires a running Qdrant container at QDRANT_URL (default
http://localhost:6333). Bring it up with:

    docker run -d --name vdbbench-qdrant-dev -p 6333:6333 -p 6334:6334 \
        qdrant/qdrant:v1.13.0
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from vdbbench.adapters.qdrant import QdrantAdapter

pytestmark = pytest.mark.integration

DEFAULT_URL = "http://localhost:6333"


def _url() -> str:
    return os.environ.get("QDRANT_URL", DEFAULT_URL)


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


@pytest.fixture
def adapter() -> Iterator[QdrantAdapter]:
    collection = f"vdbbench_test_{uuid.uuid4().hex[:8]}"
    a = QdrantAdapter(url=_url(), collection=collection)
    yield a
    with contextlib.suppress(Exception):
        a.teardown()


class TestQdrantLifecycle:
    def test_setup_idempotent(self, adapter: QdrantAdapter) -> None:
        adapter.setup(dim=8, params={})
        adapter.setup(dim=8, params={})

    def test_unknown_metric_rejected(self, adapter: QdrantAdapter) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            adapter.setup(dim=4, params={"metric": "manhattan"})

    def test_zero_knob_rejected(self, adapter: QdrantAdapter) -> None:
        with pytest.raises(ValueError, match="m must be positive"):
            adapter.setup(dim=4, params={"m": 0})


class TestQdrantIngestSearch:
    def test_round_trip(self, adapter: QdrantAdapter) -> None:
        adapter.setup(dim=16, params={"m": 8, "ef_construction": 32})
        ids = [f"p{i}" for i in range(50)]
        vecs = _random_vectors(50, 16)
        stats = adapter.ingest(ids, vecs, batch_size=10)
        assert stats.n_vectors == 50
        assert stats.elapsed_s > 0

        adapter.build_index()
        result = adapter.search(vecs[5], k=5)
        # The search should find p5 (the source vector) at or near the top.
        assert "p5" in result
        assert len(result) == 5

    def test_dim_mismatch_rejected(self, adapter: QdrantAdapter) -> None:
        adapter.setup(dim=8, params={})
        with pytest.raises(ValueError, match="dim"):
            adapter.ingest(["p0"], _random_vectors(1, 16))

    def test_search_before_setup_rejected(self, adapter: QdrantAdapter) -> None:
        with pytest.raises(RuntimeError, match="setup"):
            adapter.search(np.zeros(8, dtype=np.float32), k=5)

    def test_str_id_round_trips_through_payload(self, adapter: QdrantAdapter) -> None:
        """Qdrant uses int point ids internally; the original string id has to
        come back through the payload, not as the point id."""
        adapter.setup(dim=4, params={})
        ids = ["abc", "xyz-7", "weird:id"]
        vecs = _random_vectors(3, 4)
        adapter.ingest(ids, vecs)
        result = adapter.search(vecs[1], k=1)
        assert result == ["xyz-7"]
