"""Qdrant adapter.

Uses the standard Qdrant Python client over HTTP (port 6333) by default.
Supported `params` keys:

    {
        "metric": "cosine" | "l2" | "inner",  # default: "cosine"
        "m": int,                # HNSW M (default: 16)
        "ef_construction": int,  # HNSW ef_construction (default: 100)
        "ef_search": int,        # HNSW search-time ef (default: 64)
        "on_disk_payload": bool, # default: False
        "payload_indexed_fields": list[str],  # fields to index for filter+ANN
    }

Qdrant always uses HNSW for vector search; passing `index="none"` is
equivalent to a tiny `m` and is not exposed here on purpose.

`payload_indexed_fields` controls payload schema for hybrid filter+ANN
queries — Qdrant requires the field be indexed with a known type before
`Filter` clauses can prune the search space. The harness ingests a
`pid` payload by default; add the keys you intend to filter on.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, cast

import numpy as np

from vdbbench.adapters.base import IndexStats, IngestStats

if TYPE_CHECKING:  # pragma: no cover - imports only resolve when qdrant-client is installed
    from qdrant_client import QdrantClient

_DISTANCE_MAP: dict[str, str] = {
    "cosine": "Cosine",
    "l2": "Euclid",
    "inner": "Dot",
}


class QdrantAdapter:
    """A `VectorStoreAdapter` backed by Qdrant."""

    name = "qdrant"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "vdbbench_vectors",
        api_key: str | None = None,
    ) -> None:
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._dim: int | None = None
        self._params: dict[str, object] = {}
        self._distance: str = "Cosine"
        self._client: QdrantClient | None = None
        # Map string ids -> integer ids for Qdrant (which prefers ints).
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}

    # ---------- lifecycle ----------

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = str(params.get("metric", "cosine"))
        if metric not in _DISTANCE_MAP:
            raise ValueError(f"unknown metric {metric!r}; expected one of {list(_DISTANCE_MAP)}")
        for knob in ("m", "ef_construction", "ef_search"):
            if knob in params and int(cast(int | str, params[knob])) <= 0:
                raise ValueError(f"{knob} must be positive, got {params[knob]!r}")

        client = self._connect()
        from qdrant_client import models  # noqa: PLC0415

        # `recreate_collection` is deprecated in newer client versions; do
        # explicit delete-then-create so this works across releases.
        with contextlib.suppress(Exception):
            client.delete_collection(collection_name=self._collection)
        client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=getattr(models.Distance, _DISTANCE_MAP[metric].upper()),
                hnsw_config=models.HnswConfigDiff(
                    m=int(cast(int | str, params.get("m", 16))),
                    ef_construct=int(cast(int | str, params.get("ef_construction", 100))),
                ),
                on_disk=bool(params.get("on_disk_payload", False)),
            ),
        )

        # Pre-create payload indexes so filter+ANN queries don't fall back
        # to a linear scan of `payload`. Qdrant requires the schema to be
        # declared up front. We default each indexed field to `keyword` —
        # the right type for the harness's `pid` and any string tag.
        indexed_fields = params.get("payload_indexed_fields") or []
        if isinstance(indexed_fields, list | tuple):
            for field_name in indexed_fields:
                client.create_payload_index(
                    collection_name=self._collection,
                    field_name=str(field_name),
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

        self._dim = dim
        self._params = dict(params)
        self._distance = _DISTANCE_MAP[metric]
        self._id_to_int.clear()
        self._int_to_id.clear()

    def teardown(self) -> None:
        client = self._connect()
        with contextlib.suppress(Exception):
            client.delete_collection(collection_name=self._collection)
        self._id_to_int.clear()
        self._int_to_id.clear()

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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)

        from qdrant_client import models  # noqa: PLC0415

        client = self._connect()
        # Allocate dense integer ids in insertion order. We persist the
        # original string id in the payload so search results can map back.
        next_id = len(self._id_to_int)
        new_int_ids: list[int] = []
        for sid in ids:
            if sid not in self._id_to_int:
                self._id_to_int[sid] = next_id
                self._int_to_id[next_id] = sid
                next_id += 1
            new_int_ids.append(self._id_to_int[sid])

        start = time.perf_counter()
        for i in range(0, len(ids), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(ids)))
            points = [
                models.PointStruct(
                    id=new_int_ids[j],
                    vector=vectors[j].tolist(),
                    payload={"pid": ids[j]},
                )
                for j in range(batch_slice.start, batch_slice.stop)
            ]
            client.upsert(collection_name=self._collection, points=points, wait=True)
        elapsed = time.perf_counter() - start
        return IngestStats(n_vectors=len(ids), elapsed_s=elapsed)

    def build_index(self) -> IndexStats:
        # Qdrant builds the HNSW index incrementally during ingest; nothing
        # to do here beyond reporting the on-disk footprint of the collection.
        if self._dim is None:
            raise RuntimeError("build_index() called before setup()")
        client = self._connect()
        info = client.get_collection(collection_name=self._collection)
        # Newer versions expose `disk_data_size`; fall back to 0 for unknown.
        bytes_disk = int(getattr(info, "disk_data_size", None) or 0)
        return IndexStats(elapsed_s=0.0, bytes_disk=bytes_disk)

    # ---------- search ----------

    def search(self, query: np.ndarray, k: int) -> list[str]:
        if self._dim is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape!r}")
        if query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != setup dim {self._dim}")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)

        from qdrant_client import models  # noqa: PLC0415

        client = self._connect()
        ef = self._params.get("ef_search")
        search_params = (
            models.SearchParams(hnsw_ef=int(cast(int | str, ef))) if ef is not None else None
        )
        # `query_points` replaced the deprecated `search()` in qdrant-client
        # 1.10. Returns a `QueryResponse` with `.points` instead of a bare list.
        result = client.query_points(
            collection_name=self._collection,
            query=query.tolist(),
            limit=k,
            search_params=search_params,
            with_payload=True,
        )
        return [str(p.payload["pid"]) for p in result.points if p.payload and "pid" in p.payload]

    def memory_footprint_bytes(self) -> int:
        # Same caveat as pgvector — sample from outside (`docker stats`).
        return 0

    # ---------- internals ----------

    def _connect(self) -> QdrantClient:
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient  # noqa: PLC0415

        self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client
