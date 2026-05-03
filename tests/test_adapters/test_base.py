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
