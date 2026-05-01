"""Adapter contract test — every available adapter on a minimal corpus.

This is the safety net that catches "this adapter regressed" before the
full bench harness ever runs. The reference brute-force adapter runs on
every platform unconditionally; the real adapters (pgvector, qdrant,
lancedb, chroma) are also driven through the same contract via
parametrization, with skip-on-unavailable so the suite still passes
where the service or the wheel is missing.

The contract is:

    setup(dim, params) -> ingest(ids, vectors) -> build_index() ->
        search(query, k) returns the gold-standard ids ->
        teardown() leaves no residue.

CI gating (so the matrix is legible):

    * `exact` — runs everywhere, gated by nothing.
    * `pgvector` / `qdrant` — run on jobs with the relevant Docker
      service container (`integration-pgvector`, `integration-qdrant`).
      Locally, set ``PGVECTOR_DSN`` / ``QDRANT_URL`` to opt in; the
      fixture skips otherwise.
    * `lancedb` / `chroma` — embedded; gated only by `pytest.importorskip`,
      so they participate on Linux + arm64 macOS but skip on Intel macOS
      where the wheel is missing.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

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


def _make_exact() -> VectorStoreAdapter:
    return _ExactNNAdapter()


def _make_pgvector() -> VectorStoreAdapter:
    """Build a pgvector adapter against a service container, or skip."""
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    dsn = os.environ.get("PGVECTOR_DSN")
    if not dsn:
        pytest.skip("PGVECTOR_DSN not set; pgvector contract test needs a live service")
    # Try to actually connect so we skip cleanly when the env var is set
    # but the service is unreachable (common on dev laptops without `make up`).
    import psycopg  # noqa: PLC0415

    try:
        with psycopg.connect(dsn, connect_timeout=2) as _:
            pass
    except Exception as exc:  # pragma: no cover - service-unreachable path
        pytest.skip(f"pgvector unreachable at {dsn}: {exc}")
    from vdbbench.adapters.pgvector import PgVectorAdapter  # noqa: PLC0415

    table = f"vdbbench_contract_{uuid.uuid4().hex[:8]}"
    return PgVectorAdapter(dsn=dsn, table=table)


def _make_qdrant() -> VectorStoreAdapter:
    pytest.importorskip("qdrant_client")
    url = os.environ.get("QDRANT_URL")
    if not url:
        pytest.skip("QDRANT_URL not set; qdrant contract test needs a live service")
    # Probe the service so a misconfigured URL skips rather than failing
    # halfway through the contract.
    import httpx  # noqa: PLC0415

    try:
        httpx.get(f"{url}/healthz", timeout=2.0)
    except Exception as exc:  # pragma: no cover - service-unreachable path
        pytest.skip(f"qdrant unreachable at {url}: {exc}")
    from vdbbench.adapters.qdrant import QdrantAdapter  # noqa: PLC0415

    collection = f"vdbbench_contract_{uuid.uuid4().hex[:8]}"
    return QdrantAdapter(url=url, collection=collection)


def _make_lancedb(tmp_dir: Any) -> VectorStoreAdapter:
    pytest.importorskip("lancedb")
    from vdbbench.adapters.lancedb import LanceDBAdapter  # noqa: PLC0415

    return LanceDBAdapter(path=tmp_dir / f"lance_{uuid.uuid4().hex[:8]}")


def _make_chroma(tmp_dir: Any) -> VectorStoreAdapter:
    pytest.importorskip("chromadb")
    from vdbbench.adapters.chroma import ChromaAdapter  # noqa: PLC0415

    return ChromaAdapter(path=tmp_dir / f"chroma_{uuid.uuid4().hex[:8]}")


# Recall floor per adapter: exact NN gets 1.0, real ANN adapters can be
# slightly below 1.0 on a 64-vector corpus (HNSW with default knobs is
# essentially exact at this scale, but ivf-pq can lose a hit).
_RECALL_FLOORS: dict[str, float] = {
    "exact": 1.0,
    "pgvector": 1.0,
    "qdrant": 1.0,
    "lancedb": 0.6,
    "chroma": 1.0,
}


@pytest.fixture(
    params=["exact", "pgvector", "qdrant", "lancedb", "chroma"],
    ids=["exact", "pgvector", "qdrant", "lancedb", "chroma"],
)
def adapter(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[VectorStoreAdapter]:
    """Yield a real adapter for each parametrize id, skipping on unavailable.

    Real adapters get teardown via the fixture so a failing assertion in
    the contract still drops the table / collection / directory rather
    than leaking state into the next test.
    """
    kind = request.param
    if kind == "exact":
        a = _make_exact()
    elif kind == "pgvector":
        a = _make_pgvector()
    elif kind == "qdrant":
        a = _make_qdrant()
    elif kind == "lancedb":
        a = _make_lancedb(tmp_path)
    elif kind == "chroma":
        a = _make_chroma(tmp_path)
    else:  # pragma: no cover - unreachable
        raise ValueError(f"unknown adapter kind {kind!r}")
    try:
        yield a
    finally:
        with contextlib.suppress(Exception):
            a.teardown()


@pytest.fixture
def recall_floor(request: pytest.FixtureRequest) -> float:
    """Per-adapter recall floor for the parametrized contract test."""
    # `request.node.callspec.id` is the parametrize id; for tests that
    # depend on the `adapter` fixture this resolves to e.g. "qdrant".
    callspec = getattr(request.node, "callspec", None)
    return _RECALL_FLOORS.get(callspec.id if callspec else "exact", 1.0)


class TestAdapterContract:
    """Lifecycle invariants every adapter must satisfy."""

    def test_lifecycle_returns_correct_top1(
        self, adapter: VectorStoreAdapter, recall_floor: float
    ) -> None:
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
            assert recall >= recall_floor, (
                f"recall {recall:.2f} below floor {recall_floor:.2f} for {adapter.name}"
            )
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

    def test_recall_at_10_on_synthetic_1k(
        self, adapter: VectorStoreAdapter, recall_floor: float
    ) -> None:
        """Recall@10 must clear the per-adapter floor on a 1k synthetic corpus.

        This is the contract that catches "this adapter regressed" lifecycle
        drift before the full bench harness runs. 1k vectors is small
        enough to keep the test fast and large enough that ANN adapters
        actually exercise their indexing paths (HNSW/IVF-PQ on a 64-vector
        corpus is essentially exact and doesn't catch many regressions).
        """
        n, dim, k = 1_000, 16, 10
        rng = np.random.default_rng(7)
        passage_vecs = _l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
        query_vecs = _l2_normalize(rng.standard_normal((20, dim)).astype(np.float32))
        pids = [f"p{i:05d}" for i in range(n)]
        sims = query_vecs @ passage_vecs.T
        gold = [{pids[int(j)] for j in np.argsort(-sims[i])[:k]} for i in range(20)]

        adapter.setup(dim=dim, params={})
        try:
            adapter.ingest(pids, passage_vecs)
            adapter.build_index()
            recalls = []
            for i, gold_set in enumerate(gold):
                got = set(adapter.search(query_vecs[i], k=k))
                recalls.append(len(got & gold_set) / k)
            mean_recall = float(np.mean(recalls))
            assert mean_recall >= recall_floor, (
                f"recall@{k}={mean_recall:.3f} below floor {recall_floor:.3f} for {adapter.name}"
            )
        finally:
            adapter.teardown()
