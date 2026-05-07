"""Unit tests for the Qdrant adapter.

These exercise pure-Python contract behaviour (lifecycle state, validation
guards) without touching a live Qdrant. The integration suite at
``test_qdrant_integration.py`` covers the round-trip end-to-end against a
container; this file covers what's testable with just a stub client.
"""

from __future__ import annotations

import numpy as np
import pytest

from vdbbench.adapters.qdrant import QdrantAdapter


class _StubClient:
    """Minimal stand-in for the Qdrant client surface used by ``teardown``.

    ``delete_collection`` is suppressed-with-Exception in the adapter so
    we deliberately raise here to prove the suppression contract still
    holds (a teardown that fails to reach the server must not crash the
    Python state reset).
    """

    def __init__(self) -> None:
        self.delete_calls: list[str] = []
        self.raise_on_delete = False

    def delete_collection(self, *, collection_name: str) -> None:
        self.delete_calls.append(collection_name)
        if self.raise_on_delete:
            raise RuntimeError("simulated network failure")


class TestQdrantTeardownResetsDim:
    """Round-5 fix: ``teardown()`` must null ``_dim`` so a stray
    ``ingest()`` call after teardown raises the existing ``"called
    before setup()"`` guard instead of silently sending data to the
    just-deleted collection.

    The pgvector adapter has carried this contract since v1; the Qdrant
    adapter missed it. Asymmetry is the bug — these tests pin the
    parity.
    """

    def _adapter(self) -> tuple[QdrantAdapter, _StubClient]:
        adapter = QdrantAdapter()
        # Simulate a successful setup() without touching the network: we
        # just need ``_dim`` to be populated.
        adapter._dim = 16
        adapter._distance = "Cosine"
        adapter._params = {"m": 16}
        client = _StubClient()
        adapter._client = client  # type: ignore[assignment]
        return adapter, client

    def test_teardown_nulls_dim(self) -> None:
        adapter, _ = self._adapter()
        assert adapter._dim is not None  # precondition
        adapter.teardown()
        assert adapter._dim is None, (
            "teardown() must reset _dim so a post-teardown ingest() trips "
            "the 'called before setup()' guard rather than silently "
            "writing to the just-deleted collection"
        )

    def test_post_teardown_ingest_rejects(self) -> None:
        adapter, _ = self._adapter()
        adapter.teardown()
        with pytest.raises(RuntimeError, match="called before setup"):
            adapter.ingest(["p1"], np.zeros((1, 16), dtype=np.float32))

    def test_teardown_clears_id_maps(self) -> None:
        adapter, _ = self._adapter()
        adapter._id_to_int["p1"] = 1
        adapter._int_to_id[1] = "p1"
        adapter.teardown()
        assert adapter._id_to_int == {}
        assert adapter._int_to_id == {}

    def test_teardown_suppresses_delete_failure(self) -> None:
        # The existing contract is that a teardown can race a flaky
        # connection — the adapter swallows the delete failure so the
        # caller can still proceed to a fresh setup(). Pin it.
        adapter, client = self._adapter()
        client.raise_on_delete = True
        adapter.teardown()  # must not raise
        assert adapter._dim is None  # state still resets
