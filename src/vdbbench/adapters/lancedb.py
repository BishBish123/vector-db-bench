"""LanceDB adapter (embedded — no service).

LanceDB is the only adapter without a separate server: the storage lives in
a directory on disk, queried via the embedded `lancedb` library. That makes
it the natural baseline for "no-ops, no-network" benchmarks.

Supported `params` keys:

    {
        "metric": "cosine" | "l2" | "inner",  # default: "cosine"
        "index": "ivf_pq" | "none",            # default: "ivf_pq"
        "num_partitions": int,                  # default: sqrt(N)
        "num_sub_vectors": int,                 # default: dim/16
        "nprobes": int,                         # query-time (default: 20)
    }

LanceDB only ships wheels for Linux + arm64 macOS + Windows; this adapter
imports `lancedb` lazily so the rest of the package stays importable on
Intel macOS too.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from vdbbench.adapters.base import IndexStats, IngestStats

_DISTANCE_MAP: dict[str, str] = {
    "cosine": "cosine",
    "l2": "l2",
    "inner": "dot",
}


class LanceDBAdapter:
    """A `VectorStoreAdapter` backed by embedded LanceDB."""

    name = "lancedb"

    def __init__(
        self, path: str | Path = "./data/lancedb", table: str = "vdbbench_vectors"
    ) -> None:
        self._path = Path(path)
        self._table = table
        self._dim: int | None = None
        self._params: dict[str, object] = {}
        self._distance: str = "cosine"
        self._db: Any = None
        self._tbl: Any = None

    # ---------- lifecycle ----------

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = str(params.get("metric", "cosine"))
        if metric not in _DISTANCE_MAP:
            raise ValueError(f"unknown metric {metric!r}; expected one of {list(_DISTANCE_MAP)}")
        index_kind = str(params.get("index", "ivf_pq")).lower()
        if index_kind not in {"ivf_pq", "none"}:
            raise ValueError(f"unknown index kind {index_kind!r}; expected ivf_pq or none")
        for knob in ("num_partitions", "num_sub_vectors", "nprobes"):
            if knob in params and int(cast(int | str, params[knob])) <= 0:
                raise ValueError(f"{knob} must be positive, got {params[knob]!r}")

        import lancedb  # noqa: PLC0415  -- optional import
        import pyarrow as pa  # noqa: PLC0415

        self._path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(self._path)
        # Drop any existing table; LanceDB's drop_table raises on missing.
        with contextlib.suppress(Exception):
            db.drop_table(self._table)
        # Empty table with the right schema; ingest will append.
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        empty = pa.table({"id": [], "vector": []}, schema=schema)
        self._tbl = db.create_table(self._table, empty)
        self._db = db
        self._dim = dim
        self._params = dict(params)
        self._distance = _DISTANCE_MAP[metric]

    def teardown(self) -> None:
        if self._db is None:
            return
        with contextlib.suppress(Exception):
            self._db.drop_table(self._table)
        # Best-effort cleanup of the directory if it's now empty.
        try:
            if self._path.exists() and not any(self._path.iterdir()):
                shutil.rmtree(self._path)
        except OSError:
            pass
        self._db = None
        self._tbl = None

    # ---------- ingest ----------

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 4096) -> IngestStats:
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

        import pyarrow as pa  # noqa: PLC0415

        start = time.perf_counter()
        for i in range(0, len(ids), batch_size):
            stop = min(i + batch_size, len(ids))
            batch_vecs = vectors[i:stop]
            flat = pa.array(batch_vecs.reshape(-1), type=pa.float32())
            vector_col = pa.FixedSizeListArray.from_arrays(flat, self._dim)
            batch = pa.table({"id": pa.array(ids[i:stop], type=pa.string()), "vector": vector_col})
            self._tbl.add(batch)
        elapsed = time.perf_counter() - start
        return IngestStats(n_vectors=len(ids), elapsed_s=elapsed)

    # ---------- index ----------

    def build_index(self) -> IndexStats:
        if self._dim is None or self._tbl is None:
            raise RuntimeError("build_index() called before setup()")
        index_kind = str(self._params.get("index", "ivf_pq")).lower()

        start = time.perf_counter()
        if index_kind == "ivf_pq":
            n = self._tbl.count_rows()
            num_partitions = int(
                cast(
                    int | str,
                    self._params.get("num_partitions", max(1, int(math.sqrt(max(n, 1))))),
                )
            )
            num_sub_vectors = int(
                cast(int | str, self._params.get("num_sub_vectors", max(1, self._dim // 16)))
            )
            self._tbl.create_index(
                metric=self._distance,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        elapsed = time.perf_counter() - start
        # Sum of file sizes under the table directory.
        bytes_disk = 0
        table_path = self._path / f"{self._table}.lance"
        if table_path.exists():
            bytes_disk = sum(p.stat().st_size for p in table_path.rglob("*") if p.is_file())
        return IndexStats(elapsed_s=elapsed, bytes_disk=bytes_disk)

    # ---------- search ----------

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[str]:
        if filter is not None:
            raise NotImplementedError(
                "lancedb adapter does not support filter+ANN; pass filter=None"
            )
        if self._dim is None or self._tbl is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape!r}")
        if query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != setup dim {self._dim}")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)

        nprobes = self._params.get("nprobes")
        builder = self._tbl.search(query.tolist()).limit(k).select(["id"])
        if nprobes is not None:
            builder = builder.nprobes(int(cast(int | str, nprobes)))
        rows = builder.to_list()
        return [str(r["id"]) for r in rows]

    def memory_footprint_bytes(self) -> int:
        # Embedded — counts against the Python process; the bench harness
        # samples that from the outside via `psutil`.
        return 0
