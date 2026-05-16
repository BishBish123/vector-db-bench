"""Chroma adapter (embedded, persistent client).

Chroma is included as the "no-tuning baseline" — the brief explicitly
calls it out as the on-ramp store. We use the persistent client (storage
on disk) rather than the in-memory client so multiple bench runs can
share state and so memory measurements include the on-disk footprint.

Supported `params` keys:

    {
        "metric": "cosine" | "l2" | "inner",   # default: "cosine"
    }

Chroma doesn't expose IVF/HNSW knobs in its public API; the engine picks
the index automatically. That's part of the comparison: pgvector / qdrant /
lancedb let you tune, chroma doesn't.

`chromadb` only ships wheels for Linux + arm64 macOS + Windows (its
onnxruntime dep does not), so this adapter imports lazily.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from vdbbench.adapters.base import (
    IndexStats,
    IngestStats,
    OptionalAdapterUnavailableError,
)

_CHROMA_INSTALL_HINT = (
    "Chroma requires `pip install vdbbench[chroma]` and an importable "
    "`chromadb` (which depends on `onnxruntime`). Wheels are available for "
    "Linux x86_64, ARM64 macOS, and Windows; Intel macOS is unsupported "
    "upstream. Install instructions: https://docs.trychroma.com/getting-started"
)

_DISTANCE_MAP: dict[str, str] = {
    "cosine": "cosine",
    "l2": "l2",
    "inner": "ip",
}


class ChromaAdapter:
    """A `VectorStoreAdapter` backed by Chroma's persistent client."""

    name = "chroma"

    def __init__(
        self, path: str | Path = "./data/chroma", collection: str = "vdbbench_vectors"
    ) -> None:
        self._path = Path(path)
        self._collection_name = collection
        self._dim: int | None = None
        self._params: dict[str, object] = {}
        self._distance: str = "cosine"
        self._client: Any = None
        self._collection: Any = None

    # ---------- lifecycle ----------

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = str(params.get("metric", "cosine"))
        if metric not in _DISTANCE_MAP:
            raise ValueError(f"unknown metric {metric!r}; expected one of {list(_DISTANCE_MAP)}")

        try:
            import chromadb  # noqa: PLC0415  -- optional import
        except ImportError as exc:  # pragma: no cover - exercised in unit test via monkeypatch
            raise OptionalAdapterUnavailableError(_CHROMA_INSTALL_HINT) from exc

        self._path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._path))
        with contextlib.suppress(Exception):
            client.delete_collection(self._collection_name)
        self._collection = client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _DISTANCE_MAP[metric]},
        )
        self._client = client
        self._dim = dim
        self._params = dict(params)
        self._distance = _DISTANCE_MAP[metric]

    def teardown(self) -> None:
        if self._client is None:
            return
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._collection_name)
        try:
            if self._path.exists() and not any(self._path.iterdir()):
                shutil.rmtree(self._path)
        except OSError:
            pass
        self._collection = None
        self._client = None

    def cleanup_partial_setup(self) -> None:
        """Remove the persist_directory created by a failed ``setup()`` call.

        Called by the bench runner when ``setup()`` raises mid-way so the
        partially-written Chroma directory doesn't pollute the next run.
        ``teardown()`` is only called after a successful ``setup()``; this
        method fills the gap for interrupted setups.
        """
        try:
            if self._path.exists():
                shutil.rmtree(self._path)
        except OSError:
            pass
        self._collection = None
        self._client = None

    # ---------- ingest ----------

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1000) -> IngestStats:
        if self._dim is None or self._collection is None:
            raise RuntimeError("ingest() called before setup()")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape!r}")
        if vectors.shape[0] != len(ids):
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} ids vs {vectors.shape[0]} rows"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != setup dim {self._dim}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        start = time.perf_counter()
        for i in range(0, len(ids), batch_size):
            stop = min(i + batch_size, len(ids))
            self._collection.add(
                ids=ids[i:stop],
                embeddings=vectors[i:stop].tolist(),
            )
        elapsed = time.perf_counter() - start
        return IngestStats(n_vectors=len(ids), elapsed_s=elapsed)

    def build_index(self) -> IndexStats:
        # Chroma maintains its index incrementally; nothing to do here.
        if self._dim is None:
            raise RuntimeError("build_index() called before setup()")
        bytes_disk = 0
        if self._path.exists():
            bytes_disk = sum(p.stat().st_size for p in self._path.rglob("*") if p.is_file())
        return IndexStats(elapsed_s=0.0, bytes_disk=bytes_disk)

    # ---------- search ----------

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[str]:
        if filter is not None:
            raise NotImplementedError(
                "chroma adapter does not support filter+ANN; pass filter=None"
            )
        if self._dim is None or self._collection is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape!r}")
        if query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != setup dim {self._dim}")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)

        result = self._collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            include=[],  # only need ids
        )
        # result["ids"] is a list-of-lists keyed by query — pull the first row.
        ids = result.get("ids", [[]])
        return [str(i) for i in (ids[0] if ids else [])]

    def memory_footprint_bytes(self) -> int:
        # Embedded — counts toward the Python process; the bench harness
        # samples it externally via psutil.
        return 0
