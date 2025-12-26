"""Unit tests for the adapter base contract (no real DB)."""

from __future__ import annotations

import pytest

from vdbbench.adapters.base import IndexStats, IngestStats


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
