"""In-process brute-force adapter for smoke / debug runs.

Same shape as the production adapters (pgvector / qdrant / lancedb /
chroma) but holds vectors in a numpy matrix and answers each search
with an exact dot-product top-k. Useful for:

* CLI smoke runs that don't want a Docker dependency
  (``vdbbench bench --adapter exact``).
* The ``scripts/smoke_pipeline.py`` end-to-end check.
* Tests that need a known-correct reference: exact search has recall=1
  by construction, so a downstream regression elsewhere in the bench
  shows up as a non-1.0 recall against this adapter.

Cosine, L2, and inner-product metrics are all implemented. The adapter
is tiny on purpose — production-scale numbers come from the real
adapters; this one is the harness's own canary.
"""

from __future__ import annotations

import time

import numpy as np

from vdbbench.adapters.base import IndexStats, IngestStats

_SUPPORTED_METRICS = ("cosine", "l2", "inner")


class ExactAdapter:
    """A `VectorStoreAdapter` that answers every search with brute-force NN.

    Aliased as ``memory`` in the CLI ``--adapter`` flag because the data
    is held entirely in process RAM; both names hit this class.
    """

    name = "exact"

    def __init__(self) -> None:
        self._dim: int | None = None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._metric: str = "cosine"

    # ---------- lifecycle ----------

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = str(params.get("metric", "cosine"))
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {list(_SUPPORTED_METRICS)}"
            )
        self._dim = dim
        self._metric = metric
        self._ids = []
        self._mat = None

    def teardown(self) -> None:
        self._ids = []
        self._mat = None
        self._dim = None

    # ---------- ingest ----------

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        if self._dim is None:
            raise RuntimeError("ingest() called before setup()")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape!r}")
        if vectors.shape[0] != len(ids):
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} ids vs {vectors.shape[0]} rows"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != setup dim {self._dim}")
        start = time.perf_counter()
        self._ids = list(ids)
        # float32 keeps memory honest for large smoke runs and matches
        # what every other adapter ingests.
        self._mat = vectors.astype(np.float32, copy=False)
        elapsed = time.perf_counter() - start
        return IngestStats(n_vectors=len(ids), elapsed_s=elapsed)

    def build_index(self) -> IndexStats:
        # Brute-force: no index. Report 0 / 0 — the harness already
        # tolerates that for embedded adapters.
        return IndexStats(elapsed_s=0.0, bytes_disk=0)

    # ---------- search ----------

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[str]:
        if filter is not None:
            raise NotImplementedError("ExactAdapter does not implement payload filters")
        if self._dim is None or self._mat is None:
            raise RuntimeError("search() called before ingest()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape!r}")
        if query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != setup dim {self._dim}")

        q = query.astype(np.float32, copy=False)
        if self._metric == "cosine":
            # Normalise both sides — cheaper to L2-normalise the matrix
            # once on the fly than to re-divide every search.
            q_norm = q / (np.linalg.norm(q) or 1.0)
            mat_norms = np.linalg.norm(self._mat, axis=1)
            mat_norms[mat_norms == 0] = 1.0
            sims = (self._mat @ q_norm) / mat_norms
            top_idx = np.argsort(-sims)[:k]
        elif self._metric == "inner":
            sims = self._mat @ q
            top_idx = np.argsort(-sims)[:k]
        else:  # l2 — ascending distance
            diffs = self._mat - q
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            top_idx = np.argsort(d2)[:k]
        return [self._ids[i] for i in top_idx]

    def memory_footprint_bytes(self) -> int:
        if self._mat is None:
            return 0
        return int(self._mat.nbytes)
