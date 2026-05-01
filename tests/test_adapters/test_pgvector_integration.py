"""Integration tests for the pgvector adapter.

Requires a running pgvector container at PGVECTOR_DSN (default
postgresql://bench:bench@localhost:5433/bench). Marked `integration` so
they are skipped on developer machines without Docker.

Bring the container up with:

    docker run -d --name vdbbench-pg-dev -e POSTGRES_PASSWORD=bench \
        -e POSTGRES_USER=bench -e POSTGRES_DB=bench -p 5433:5432 \
        pgvector/pgvector:pg17
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from vdbbench.adapters.pgvector import PgVectorAdapter

pytestmark = pytest.mark.integration

DEFAULT_DSN = "postgresql://bench:bench@localhost:5433/bench"


def _dsn() -> str:
    return os.environ.get("PGVECTOR_DSN", DEFAULT_DSN)


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


@pytest.fixture
def adapter() -> Iterator[PgVectorAdapter]:
    """Fresh adapter against an isolated table per test."""
    table = f"vdbbench_test_{uuid.uuid4().hex[:8]}"
    a = PgVectorAdapter(dsn=_dsn(), table=table)
    yield a
    with contextlib.suppress(Exception):
        a.teardown()


class TestPgVectorLifecycle:
    def test_setup_creates_table(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(dim=8, params={"index": "none"})
        # Idempotency: a second setup wipes and recreates the table.
        adapter.setup(dim=8, params={"index": "none"})

    def test_unknown_metric_rejected(self, adapter: PgVectorAdapter) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            adapter.setup(dim=4, params={"metric": "manhattan"})

    def test_zero_dim_rejected(self, adapter: PgVectorAdapter) -> None:
        with pytest.raises(ValueError, match="dim must be positive"):
            adapter.setup(dim=0, params={})

    def test_unknown_index_kind_rejected_in_setup(self, adapter: PgVectorAdapter) -> None:
        with pytest.raises(ValueError, match="unknown index kind"):
            adapter.setup(dim=4, params={"index": "ivf"})  # typo, should be ivfflat

    def test_zero_knob_rejected_in_setup(self, adapter: PgVectorAdapter) -> None:
        with pytest.raises(ValueError, match="probes must be positive"):
            adapter.setup(dim=4, params={"index": "ivfflat", "probes": 0})

    def test_state_clean_after_failed_setup(self, adapter: PgVectorAdapter) -> None:
        """A rejected setup must not leave the adapter looking initialized."""
        with pytest.raises(ValueError):
            adapter.setup(dim=4, params={"metric": "manhattan"})
        # Subsequent ingest must report the un-setup state, not a phantom dim.
        with pytest.raises(RuntimeError, match="setup"):
            adapter.ingest(["p0"], _random_vectors(1, 4))


class TestPgVectorIngest:
    def test_round_trip_no_index(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(dim=16, params={"index": "none"})
        ids = [f"p{i}" for i in range(50)]
        vecs = _random_vectors(50, 16)
        stats = adapter.ingest(ids, vecs, batch_size=10)
        assert stats.n_vectors == 50
        assert stats.elapsed_s > 0
        assert stats.throughput_vps > 0

    def test_search_returns_self_first(self, adapter: PgVectorAdapter) -> None:
        """Searching for a passage's own vector should return that pid first."""
        adapter.setup(dim=8, params={"index": "none"})
        ids = [f"p{i}" for i in range(20)]
        vecs = _random_vectors(20, 8, seed=3)
        adapter.ingest(ids, vecs)
        # Search with the third vector — pgvector's exact distance should
        # rank it #1 (cosine of unit vec with itself = 1, distance = 0).
        result = adapter.search(vecs[3], k=5)
        assert result[0] == "p3"
        assert len(result) == 5

    def test_dim_mismatch_rejected(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(dim=8, params={"index": "none"})
        with pytest.raises(ValueError, match="dim"):
            adapter.ingest(["p0"], _random_vectors(1, 16))

    def test_id_count_mismatch_rejected(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(dim=4, params={"index": "none"})
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.ingest(["p0", "p1"], _random_vectors(3, 4))


class TestPgVectorIndex:
    def test_hnsw_index_round_trip(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(
            dim=16,
            params={
                "index": "hnsw",
                "metric": "cosine",
                "m": 8,
                "ef_construction": 32,
                "ef_search": 32,
            },
        )
        ids = [f"p{i}" for i in range(100)]
        vecs = _random_vectors(100, 16, seed=7)
        adapter.ingest(ids, vecs)
        idx = adapter.build_index()
        assert idx.elapsed_s > 0
        assert idx.bytes_disk > 0

        # Sanity recall: searching for a known vector should still rank it
        # near the top with HNSW.
        result = adapter.search(vecs[42], k=5)
        assert "p42" in result

    def test_ivfflat_index_round_trip(self, adapter: PgVectorAdapter) -> None:
        adapter.setup(
            dim=16,
            params={"index": "ivfflat", "metric": "cosine", "lists": 4, "probes": 4},
        )
        ids = [f"p{i}" for i in range(50)]
        vecs = _random_vectors(50, 16, seed=11)
        adapter.ingest(ids, vecs)
        idx = adapter.build_index()
        assert idx.elapsed_s >= 0
        result = adapter.search(vecs[7], k=5)
        assert "p7" in result

    def test_search_before_setup_rejected(self, adapter: PgVectorAdapter) -> None:
        with pytest.raises(RuntimeError, match="setup"):
            adapter.search(np.zeros(8, dtype=np.float32), k=5)


class TestPgVectorConnectionLifecycle:
    def test_one_connection_open_per_setup_not_per_query(self, adapter: PgVectorAdapter) -> None:
        """Connection-establishment should not be counted as ANN latency.

        Earlier revisions of this adapter opened a fresh psycopg connection
        on every search() call. On macOS Docker that's ~40 ms of TCP+auth
        handshake per query, which makes the bench measure connection
        churn rather than vector search. Assert exactly one open per
        setup() so a regression here fails loud.
        """
        adapter.setup(dim=8, params={"index": "none"})
        baseline = adapter._connection_opens
        ids = [f"p{i}" for i in range(20)]
        vecs = _random_vectors(20, 8, seed=5)
        adapter.ingest(ids, vecs)
        adapter.build_index()
        for i in range(50):
            adapter.search(vecs[i % 20], k=3)
        # No new connection opens across ingest + index + 50 searches.
        assert adapter._connection_opens == baseline
