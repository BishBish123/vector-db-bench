"""pgvector adapter.

Uses Postgres 17 + pgvector extension via psycopg3 + the `pgvector` Python
binding. Knobs land in the `params` dict — supported keys:

    {
        "index": "hnsw" | "ivfflat" | "none",   # default: "hnsw"
        "metric": "cosine" | "l2" | "inner",    # default: "cosine"
        "m": int,                # HNSW param (default: 16)
        "ef_construction": int,  # HNSW param (default: 64)
        "ef_search": int,        # HNSW query-time param (default: 40)
        "lists": int,            # IVFFLAT param (default: sqrt(N))
        "probes": int,           # IVFFLAT query-time param (default: 10)
        "analyze_after_index": bool,  # default: True
    }

`analyze_after_index=True` runs `ANALYZE` after the index build so the
planner has accurate statistics. Disable it for "first-query cold" timings
where you specifically want to measure plan-cache misses.

Connection lifecycle: a single psycopg connection is opened in
``setup()`` (with the ``vector`` type adapter registered once) and reused
for every ``ingest`` / ``build_index`` / ``search`` call until
``teardown()`` closes it. Earlier revisions opened a fresh connection
per ``search()``, which on macOS Docker meant we benchmarked
~40 ms of TCP+auth handshake on every query rather than ANN latency.
"""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from vdbbench.adapters.base import IndexStats, IngestStats

_DISTANCE_OPS: dict[str, tuple[str, str]] = {
    # metric -> (operator, opclass)
    "cosine": ("<=>", "vector_cosine_ops"),
    "l2": ("<->", "vector_l2_ops"),
    "inner": ("<#>", "vector_ip_ops"),
}


# Postgres identifier rule we accept: ASCII letters/digits/underscore,
# starts with a letter or underscore. Looser than the actual Postgres
# spec (which allows quoted Unicode), but stops the obvious SQL-injection
# vector from a caller who builds a `table_name` from untrusted input.
_PG_IDENT_RE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Postgres caps identifiers at NAMEDATALEN-1 = 63 bytes (UTF-8). Beyond
# that it silently truncates — which would let two distinct per-run table
# names collapse onto the same on-disk table and corrupt each other's
# results, the exact failure mode the uuid suffix exists to prevent. We
# enforce the limit here rather than let Postgres truncate quietly.
_PG_IDENT_MAX_BYTES = 63


def _validate_pg_identifier(name: str) -> None:
    if not isinstance(name, str) or not _PG_IDENT_RE.match(name):
        raise ValueError(
            f"invalid pg identifier {name!r}; expected ^[A-Za-z_][A-Za-z0-9_]*$"
        )
    encoded_len = len(name.encode("utf-8"))
    if encoded_len > _PG_IDENT_MAX_BYTES:
        raise ValueError(
            f"invalid pg identifier {name!r}: {encoded_len} bytes exceeds "
            f"Postgres NAMEDATALEN-1 limit of {_PG_IDENT_MAX_BYTES} bytes"
        )


class PgVectorAdapter:
    """A `VectorStoreAdapter` backed by Postgres + pgvector."""

    name = "pgvector"

    def __init__(self, dsn: str, table: str = "vdbbench_vectors") -> None:
        self._dsn = dsn
        self._table = table
        # Validate the constructor-time name with the same rule as
        # `set_table_name` so the two paths can't disagree.
        _validate_pg_identifier(table)
        self._dim: int | None = None
        self._params: dict[str, object] = {}
        self._operator: str = "<=>"
        self._opclass: str = "vector_cosine_ops"
        # Long-lived connection — opened in setup, closed in teardown.
        # Type set to Any so we don't require psycopg at import time.
        self._conn: Any = None
        # Counter for tests / observability: how many psycopg.connect calls
        # the adapter has made over its lifetime. With a long-lived
        # connection this should never exceed `setup() + teardown()` for a
        # single bench run.
        self._connection_opens: int = 0

    # ---------- lifecycle ----------

    def set_table_name(self, name: str) -> None:
        """Override the per-run table name before ``setup()``.

        The bench runner calls this with a per-spec, uuid-suffixed name
        so two ``run_bench`` calls against the same Postgres can't share
        a table and silently corrupt each other's results. Validated
        with the same identifier rule the constructor uses; safe to
        call before setup, undefined after.
        """
        _validate_pg_identifier(name)
        self._table = name

    def setup(self, dim: int, params: dict[str, object]) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = str(params.get("metric", "cosine"))
        if metric not in _DISTANCE_OPS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {list(_DISTANCE_OPS)}")
        # Validate every ANN knob now so a typo doesn't fail after a 100k
        # ingest. Defaults match Postgres / pgvector defaults.
        index_kind = str(params.get("index", "hnsw")).lower()
        if index_kind not in {"hnsw", "ivfflat", "none"}:
            raise ValueError(f"unknown index kind {index_kind!r}")
        for knob in ("m", "ef_construction", "ef_search", "lists", "probes"):
            if knob in params and int(cast(int | str, params[knob])) <= 0:
                raise ValueError(f"{knob} must be positive, got {params[knob]!r}")

        # If a previous setup call left a connection lying around (e.g.
        # tests reusing the same adapter), close it before opening a fresh
        # one — otherwise `setup` would not be idempotent.
        self._close_connection()
        # Open the long-lived connection once and register the vector type
        # adapter on it. Every subsequent ingest/build_index/search call
        # reuses this connection, so we measure ANN latency rather than
        # TCP+auth handshake cost.
        from pgvector.psycopg import register_vector  # noqa: PLC0415

        # Hold the freshly opened connection in a local until *all*
        # initialisation succeeds — only then "promote" it onto self._conn.
        # Until promotion, any failure path explicitly closes the local so
        # we never leak a TCP connection to the pgvector container. This
        # specifically covers register_vector() raising (e.g. when the
        # `vector` extension isn't installed yet) — earlier revisions
        # opened the connection, called register_vector, and only set up
        # exception handling for the DDL block, leaking on the
        # register_vector path.
        conn = self._open_connection()
        try:
            # Order matters: `CREATE EXTENSION IF NOT EXISTS vector` must run
            # BEFORE register_vector is called, otherwise psycopg has no
            # `vector` type to bind on the connection and register_vector
            # fails with "type 'vector' not found" against a freshly
            # initialised database where the extension hasn't been created
            # yet. With this ordering, register_vector still raises (and the
            # except below still cleans up) if the extension is missing AND
            # CREATE EXTENSION couldn't be granted — but the common
            # "fresh-DB" path now succeeds without operator intervention.
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{self._table}"')
                cur.execute(
                    f'CREATE TABLE "{self._table}" (id text PRIMARY KEY, vec vector({dim}))'
                )
                conn.commit()
        except Exception:
            # Anything during register_vector or DDL means the adapter is
            # not initialised — close the local connection (self._conn is
            # still None at this point) and propagate.
            with contextlib.suppress(Exception):
                conn.close()
            raise
        self._conn = conn
        self._dim = dim
        self._params = dict(params)
        self._operator, self._opclass = _DISTANCE_OPS[metric]

    def teardown(self) -> None:
        # Drop the table on the long-lived connection if it's still alive,
        # then close. Tolerate teardown being called twice in a row (the
        # contract test exercises that path).
        if self._conn is not None:
            drop_ok = self._attempt_drop()
            if not drop_ok:
                # First attempt failed — open a fresh connection and retry
                # once. This covers the common case where the long-lived
                # connection has gone stale (e.g. pgvector container
                # restarted mid-run) but the table still exists.
                self._close_connection()
                try:
                    conn = self._open_connection()
                    try:
                        with conn.cursor() as cur:
                            cur.execute(f'DROP TABLE IF EXISTS "{self._table}"')
                            conn.commit()
                    finally:
                        with contextlib.suppress(Exception):
                            conn.close()
                except Exception as retry_exc:
                    # Both attempts failed — table is orphaned. Log a
                    # structured warning with the table name and a ready-to-run
                    # psql cleanup command so an operator can recover manually.
                    import logging as _logging  # noqa: PLC0415

                    _log = _logging.getLogger(__name__)
                    cleanup_cmd = (
                        f"psql {self._dsn!r} -c 'DROP TABLE IF EXISTS \"{self._table}\";'"
                    )
                    _log.warning(
                        "pgvector teardown: orphan table left behind",
                        extra={
                            "orphan_table": self._table,
                            "dsn": self._dsn,
                            "cleanup_cmd": cleanup_cmd,
                            "retry_error": str(retry_exc),
                        },
                    )
                    self._append_orphan_record(cleanup_cmd)
        self._close_connection()
        self._dim = None

    def _attempt_drop(self) -> bool:
        """Try to drop the bench table on the existing connection.

        Returns ``True`` on success, ``False`` if any exception was raised.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{self._table}"')
                self._conn.commit()
            return True
        except Exception:
            return False

    def _append_orphan_record(self, cleanup_cmd: str) -> None:
        """Append orphan table info to ``results/<run>/orphan_tables.txt``.

        Best-effort — if the file can't be written the warning log is the
        primary record. Kept in the current working directory so CI runs
        and local bench runs naturally land in the right place.
        """
        import os  # noqa: PLC0415

        orphan_dir = Path(os.getcwd()) / "results"
        try:
            orphan_dir.mkdir(parents=True, exist_ok=True)
            record = (
                f"orphan_table={self._table}\tdsn={self._dsn}\tcleanup={cleanup_cmd}\n"
            )
            with open(orphan_dir / "orphan_tables.txt", "a") as fh:
                fh.write(record)
        except OSError:
            pass  # best-effort; warning log is the primary record

    # ---------- ingest ----------

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        if self._dim is None or self._conn is None:
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

        # `register_vector` already ran in setup(); the long-lived
        # connection still has the type adapter cached so we can pass
        # ndarray rows directly.
        start = time.perf_counter()
        with self._conn.cursor() as cur:
            insert_sql = f'INSERT INTO "{self._table}" (id, vec) VALUES (%s, %s)'
            for i in range(0, len(ids), batch_size):
                rows = [(ids[j], vectors[j]) for j in range(i, min(i + batch_size, len(ids)))]
                cur.executemany(insert_sql, rows)
            self._conn.commit()
        elapsed = time.perf_counter() - start
        return IngestStats(n_vectors=len(ids), elapsed_s=elapsed)

    # ---------- index ----------

    def build_index(self) -> IndexStats:
        if self._dim is None or self._conn is None:
            raise RuntimeError("build_index() called before setup()")
        # Already validated in setup; this is just to read the value.
        index_kind = str(self._params.get("index", "hnsw")).lower()

        start = time.perf_counter()
        bytes_disk = 0
        with self._conn.cursor() as cur:
            if index_kind == "none":
                pass
            elif index_kind == "hnsw":
                m = int(cast(int | str, self._params.get("m", 16)))
                ef = int(cast(int | str, self._params.get("ef_construction", 64)))
                cur.execute(
                    f'CREATE INDEX ON "{self._table}" USING hnsw '
                    f"(vec {self._opclass}) WITH (m = {m}, ef_construction = {ef})"
                )
            elif index_kind == "ivfflat":
                cur.execute(f'SELECT count(*) FROM "{self._table}"')
                row = cur.fetchone()
                n = int(row[0]) if row else 0
                lists = int(
                    cast(int | str, self._params.get("lists", max(1, int(math.sqrt(max(n, 1))))))
                )
                cur.execute(
                    f'CREATE INDEX ON "{self._table}" USING ivfflat '
                    f"(vec {self._opclass}) WITH (lists = {lists})"
                )
            if bool(self._params.get("analyze_after_index", True)):
                cur.execute(f'ANALYZE "{self._table}"')
            # Pass the *quoted* identifier into pg_total_relation_size so
            # mixed-case table names round-trip (regclass folds bare names
            # to lowercase, which would error on `BenchVectors`-style ids).
            cur.execute(
                "SELECT pg_total_relation_size(quote_ident(%s)::regclass)",
                (self._table,),
            )
            row = cur.fetchone()
            bytes_disk = int(row[0]) if row else 0
            self._conn.commit()
        elapsed = time.perf_counter() - start
        return IndexStats(elapsed_s=elapsed, bytes_disk=bytes_disk)

    # ---------- search ----------

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[str]:
        # pgvector adapter does not implement payload filters yet; the
        # `filter` arg exists only to satisfy the protocol surface so the
        # bench harness can drive every adapter through the same call.
        if filter is not None:
            raise NotImplementedError(
                "pgvector adapter does not support filter+ANN; pass filter=None"
            )
        if self._dim is None or self._conn is None:
            raise RuntimeError("search() called before setup()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape!r}")
        if query.shape[0] != self._dim:
            raise ValueError(f"query dim {query.shape[0]} != setup dim {self._dim}")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)

        # Reuse the long-lived connection — every search() used to spin up a
        # fresh psycopg.connect (TCP+auth = ~40 ms on Docker for Mac), which
        # turned the bench into a connection-establishment benchmark rather
        # than an ANN-latency benchmark.
        with self._conn.cursor() as cur:
            self._apply_query_knobs(cur)
            cur.execute(
                f'SELECT id FROM "{self._table}" ORDER BY vec {self._operator} %s LIMIT %s',
                (query, k),
            )
            return [row[0] for row in cur.fetchall()]

    def memory_footprint_bytes(self) -> int:
        # Server-side resident memory across the whole Postgres process is
        # not meaningful per-table. The bench harness samples this from the
        # outside (e.g. `docker stats`) instead.
        return 0

    def cleanup_partial_setup(self) -> None:
        # pgvector is a service adapter — no filesystem state is written by
        # setup(), so there is nothing to remove on failure.
        #
        # Callers might wonder: "shouldn't we call teardown() to DROP TABLE?"
        # The bench runner deliberately avoids that — teardown() assumes a
        # completed setup() (i.e. self._conn is non-None).  Calling it after a
        # failed setup() would be undefined behaviour.  The partial table, if
        # any, is handled by the next setup() call's own DROP TABLE IF EXISTS.
        return

    # ---------- internals ----------

    def _open_connection(self) -> Any:
        """Open a fresh psycopg connection and bump the open counter.

        Tests assert this counter to verify the long-lived-connection
        invariant (one open per `setup()`, not one per `search()`).
        """
        import psycopg  # noqa: PLC0415

        self._connection_opens += 1
        return psycopg.connect(self._dsn)

    def _close_connection(self) -> None:
        if self._conn is None:
            return
        with contextlib.suppress(Exception):
            self._conn.close()
        self._conn = None

    def _apply_query_knobs(self, cur: Any) -> None:
        """Set per-session query-time knobs (HNSW ef_search, IVF probes)."""
        index_kind = str(self._params.get("index", "hnsw")).lower()
        if index_kind == "hnsw" and "ef_search" in self._params:
            ef = int(cast(int | str, self._params["ef_search"]))
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
        elif index_kind == "ivfflat" and "probes" in self._params:
            probes = int(cast(int | str, self._params["probes"]))
            cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
