"""Integration tests for the Chroma adapter.

Skips on platforms where chromadb has no wheel (Intel macOS).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip(
    "chromadb",
    reason="chromadb has no wheel for this platform (e.g. Intel macOS); "
    "run on Linux / arm64 macOS / Windows.",
)

from vdbbench.adapters.chroma import ChromaAdapter  # noqa: E402

pytestmark = pytest.mark.integration


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[ChromaAdapter]:
    a = ChromaAdapter(path=tmp_path / f"chroma_{uuid.uuid4().hex[:8]}")
    yield a
    with contextlib.suppress(Exception):
        a.teardown()


class TestChromaLifecycle:
    def test_setup_idempotent(self, adapter: ChromaAdapter) -> None:
        adapter.setup(dim=8, params={})
        adapter.setup(dim=8, params={})

    def test_unknown_metric_rejected(self, adapter: ChromaAdapter) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            adapter.setup(dim=4, params={"metric": "manhattan"})


class TestChromaIngestSearch:
    def test_round_trip(self, adapter: ChromaAdapter) -> None:
        adapter.setup(dim=16, params={})
        ids = [f"p{i}" for i in range(50)]
        vecs = _random_vectors(50, 16)
        stats = adapter.ingest(ids, vecs, batch_size=10)
        assert stats.n_vectors == 50
        result = adapter.search(vecs[5], k=5)
        assert "p5" in result
        assert len(result) == 5

    def test_dim_mismatch_rejected(self, adapter: ChromaAdapter) -> None:
        adapter.setup(dim=8, params={})
        with pytest.raises(ValueError, match="dim"):
            adapter.ingest(["p0"], _random_vectors(1, 16))

    def test_search_before_setup_rejected(self, adapter: ChromaAdapter) -> None:
        with pytest.raises(RuntimeError, match="setup"):
            adapter.search(np.zeros(8, dtype=np.float32), k=5)
