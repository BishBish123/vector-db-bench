"""Unit tests for the adapter base contract (no real DB)."""

from __future__ import annotations

import builtins

import pytest

from vdbbench.adapters.base import (
    IndexStats,
    IngestStats,
    OptionalAdapterUnavailableError,
)


class TestIngestStats:
    def test_throughput_basic(self) -> None:
        s = IngestStats(n_vectors=100, elapsed_s=2.0)
        assert s.throughput_vps == 50.0

    def test_throughput_zero_duration(self) -> None:
        s = IngestStats(n_vectors=10, elapsed_s=0.0)
        assert s.throughput_vps == 0.0

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            IngestStats(n_vectors=-1, elapsed_s=1.0)

    def test_negative_elapsed_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            IngestStats(n_vectors=1, elapsed_s=-0.1)


class TestIndexStats:
    def test_default_disk_zero(self) -> None:
        assert IndexStats(elapsed_s=1.0).bytes_disk == 0

    def test_negative_elapsed_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            IndexStats(elapsed_s=-1.0)

    def test_negative_disk_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            IndexStats(elapsed_s=1.0, bytes_disk=-1)


class TestOptionalAdapterUnavailable:
    def test_inherits_from_import_error(self) -> None:
        # Existing `except ImportError` handlers must keep catching this.
        assert issubclass(OptionalAdapterUnavailableError, ImportError)

    def test_lancedb_setup_raises_with_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When `lancedb` isn't importable, setup() raises a custom error
        whose message points at the supported install path. Without this
        wrapper users got a bare ModuleNotFoundError naming a missing C
        extension instead of the supported install instructions.
        """
        # Imported inside the test so the patched __import__ above runs
        # against the optional `lancedb` import, not against the test's own
        # imports at module load.
        from vdbbench.adapters.lancedb import LanceDBAdapter  # noqa: PLC0415

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name in ("lancedb", "pyarrow") and level == 0:
                raise ImportError(f"No module named {name!r}")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        adapter = LanceDBAdapter(path="/tmp/nope-lancedb")
        with pytest.raises(OptionalAdapterUnavailableError, match="LanceDB"):
            adapter.setup(dim=4, params={})

    def test_chroma_setup_raises_with_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from vdbbench.adapters.chroma import ChromaAdapter  # noqa: PLC0415

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "chromadb" and level == 0:
                raise ImportError("No module named 'chromadb'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        adapter = ChromaAdapter(path="/tmp/nope-chroma")
        with pytest.raises(OptionalAdapterUnavailableError, match="Chroma"):
            adapter.setup(dim=4, params={})


class TestPgVectorSetupConnectionLifecycle:
    """`setup()` must not leak the freshly opened psycopg connection if any
    initialisation step (register_vector, DDL) raises. Earlier the
    register_vector call sat outside the try/except so a failure there
    skipped the close path entirely.
    """

    def test_setup_closes_connection_on_register_vector_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("pgvector")
        from vdbbench.adapters.pgvector import PgVectorAdapter  # noqa: PLC0415

        adapter = PgVectorAdapter(dsn="postgresql://example/none-of-this-is-real")

        # Track the synthetic connection so we can assert it was closed
        # exactly once and never promoted onto self._conn.
        class _FakeConn:
            def __init__(self) -> None:
                self.closed = 0

            def close(self) -> None:
                self.closed += 1

            def cursor(self) -> object:  # pragma: no cover - never reached
                raise AssertionError("cursor() should not run after register_vector raises")

        fake = _FakeConn()
        monkeypatch.setattr(PgVectorAdapter, "_open_connection", lambda self: fake)

        # Inject a register_vector that raises so setup() takes the
        # cleanup path. The adapter imports it lazily inside setup() so we
        # patch the module attribute the same way pgvector exposes it.
        import pgvector.psycopg as pgv_psycopg  # noqa: PLC0415

        def boom(_conn: object) -> None:
            raise RuntimeError("register_vector failure")

        monkeypatch.setattr(pgv_psycopg, "register_vector", boom)

        with pytest.raises(RuntimeError, match="register_vector failure"):
            adapter.setup(dim=4, params={})

        # Connection got closed and was never promoted.
        assert fake.closed == 1
        assert adapter._conn is None
        # And the adapter is still un-initialised so a follow-up ingest
        # call raises the un-setup error rather than a phantom-state one.
        assert adapter._dim is None


class TestPgVectorTableNameValidation:
    """``set_table_name`` is called by the bench runner with a per-run
    uuid-suffixed name; validate the rule rejects obvious injection."""

    def test_constructor_rejects_invalid_table(self) -> None:
        pytest.importorskip("pgvector")
        from vdbbench.adapters.pgvector import PgVectorAdapter  # noqa: PLC0415

        with pytest.raises(ValueError, match="invalid pg identifier"):
            PgVectorAdapter(dsn="postgresql://example/none", table='evil"; DROP TABLE x; --')

    def test_set_table_name_rejects_invalid(self) -> None:
        pytest.importorskip("pgvector")
        from vdbbench.adapters.pgvector import PgVectorAdapter  # noqa: PLC0415

        adapter = PgVectorAdapter(dsn="postgresql://example/none")
        with pytest.raises(ValueError, match="invalid pg identifier"):
            adapter.set_table_name("a b")  # space disallowed
        with pytest.raises(ValueError, match="invalid pg identifier"):
            adapter.set_table_name("1table")  # leading digit disallowed

    def test_set_table_name_accepts_uuid_suffixed_name(self) -> None:
        pytest.importorskip("pgvector")
        from vdbbench.adapters.pgvector import PgVectorAdapter  # noqa: PLC0415

        adapter = PgVectorAdapter(dsn="postgresql://example/none")
        adapter.set_table_name("vdbbench_vectors_deadbeef")
        assert adapter._table == "vdbbench_vectors_deadbeef"
