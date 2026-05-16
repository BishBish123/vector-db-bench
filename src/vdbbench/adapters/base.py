"""Adapter Protocol and shared dataclasses.

Every vector store backend (pgvector, qdrant, lancedb, chroma) implements
the same surface so the bench harness stays adapter-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class OptionalAdapterUnavailableError(ImportError):
    """An optional adapter's backing wheel is not importable on this host.

    Raised by the lazy-import path inside ``LanceDBAdapter`` /
    ``ChromaAdapter`` when the underlying package can't be loaded
    (typically Intel macOS, where neither ships a wheel). The message
    points at the relevant install instructions so the user knows what
    to do — earlier revisions surfaced a bare ``ModuleNotFoundError``
    that named the missing C extension instead of the supported install
    path.

    Inherits from ``ImportError`` so existing ``except ImportError``
    handlers still catch it without being rewritten.
    """


@dataclass(frozen=True)
class IngestStats:
    """How long the bulk-load step took, plus how many vectors landed."""

    n_vectors: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.n_vectors < 0:
            raise ValueError("n_vectors must be non-negative")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")

    @property
    def throughput_vps(self) -> float:
        """Vectors per second. Defined as 0 for zero-duration ingests."""
        if self.elapsed_s <= 0:
            return 0.0
        return self.n_vectors / self.elapsed_s


@dataclass(frozen=True)
class IndexStats:
    """How long the post-ingest index build took, plus on-disk footprint.

    `bytes_disk` may be 0 if the backend is embedded and doesn't expose a
    distinct on-disk index (LanceDB, Chroma).
    """

    elapsed_s: float
    bytes_disk: int = 0

    def __post_init__(self) -> None:
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")
        if self.bytes_disk < 0:
            raise ValueError("bytes_disk must be non-negative")


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """A single vector store backend, exposed via a uniform lifecycle.

    The bench harness drives every adapter through the exact same calls
    so per-DB tuning differences live entirely inside the `params` dict
    that gets passed to `setup`. Per-adapter knob discovery is the bench
    runner's responsibility, not the harness's.

    Lifecycle ordering:
        adapter.setup(dim, params)
        adapter.ingest(ids, vectors)
        adapter.build_index()
        for q in queries: adapter.search(q, k)
        adapter.teardown()
    """

    name: str

    def setup(self, dim: int, params: dict[str, object]) -> None:
        """Provision a fresh, empty store of dimension `dim`.

        Must be idempotent against a fresh container — calling `setup`
        twice in a row should not leave residue from the first call.
        """
        ...

    def teardown(self) -> None:
        """Drop the table / collection / index this adapter created."""
        ...

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        """Bulk-load vectors with their string ids. Vectors must be `(N, dim)` float32."""
        ...

    def build_index(self) -> IndexStats:
        """Build the ANN index after ingest. May be a no-op for embedded stores."""
        ...

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[str]:
        """Return the top-`k` ids for a single query vector (1-D).

        Adapters that support backend-side payload / metadata filters
        (currently Qdrant) accept an optional `filter` dict and pass it
        straight through. Adapters without filter support ignore the
        argument; the bench runner only ever calls `search(query, k)` so
        the default ``None`` keeps the existing surface intact.
        """
        ...

    def memory_footprint_bytes(self) -> int:
        """Resident memory of the search server, in bytes (0 if not measurable)."""
        ...

    def cleanup_partial_setup(self) -> None:
        """Remove any on-disk state created by a ``setup()`` that was interrupted.

        Called by the bench runner when ``setup()`` raises before completing
        the full lifecycle. Embedded adapters (LanceDB, Chroma) override this
        to ``rmtree`` their data directories so partial state doesn't leak
        across runs. Service adapters (pgvector, Qdrant) do not override —
        their existing ``teardown()`` paths already handle partial state, and
        the runner only calls ``teardown()`` after a successful ``setup()``.

        The default is a no-op so implementing the method is optional for
        adapters that don't need it.
        """
        return  # no-op default
