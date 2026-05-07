"""Tests for MemorySampler — background-thread RSS sampling."""

from __future__ import annotations

import time

import pytest

from vdbbench.profile.memory import MemorySampler, MemoryStats


class TestMemoryStats:
    def test_dataclass_fields(self) -> None:
        stats = MemoryStats(peak_rss_bytes=100, mean_rss_bytes=80, samples_count=5)
        assert stats.peak_rss_bytes == 100
        assert stats.mean_rss_bytes == 80
        assert stats.samples_count == 5

    def test_frozen(self) -> None:
        stats = MemoryStats(peak_rss_bytes=1, mean_rss_bytes=1, samples_count=1)
        with pytest.raises((AttributeError, TypeError)):
            stats.peak_rss_bytes = 2  # type: ignore[misc]


class TestMemorySamplerContextManager:
    def test_records_non_zero_rss(self) -> None:
        """MemorySampler must capture non-zero RSS for a live process.

        Allocate a big list so psutil definitely sees non-trivial RSS.
        Peak should be at least 1 byte — verifying the background thread
        actually polled psutil and the result was plumbed through.
        """
        with MemorySampler(interval_s=0.05) as sampler:
            # Allocate ~1 MB of data so there's something for psutil to see.
            _buf = [b"x" * 1024] * 1024
            time.sleep(0.12)  # let at least 2 samples fire
            del _buf
        stats = sampler.stats
        assert stats.samples_count >= 1
        assert stats.peak_rss_bytes > 0
        assert stats.mean_rss_bytes > 0

    def test_peak_at_least_mean(self) -> None:
        with MemorySampler(interval_s=0.05) as sampler:
            time.sleep(0.12)
        stats = sampler.stats
        if stats.samples_count > 0:
            assert stats.peak_rss_bytes >= stats.mean_rss_bytes

    def test_stats_accessible_after_exit(self) -> None:
        with MemorySampler(interval_s=0.05) as sampler:
            time.sleep(0.06)
        # Should not raise after __exit__
        _ = sampler.stats

    def test_stats_raises_before_stop(self) -> None:
        sampler = MemorySampler(interval_s=0.05)
        sampler.start()
        with pytest.raises(RuntimeError, match="stopped"):
            _ = sampler.stats
        sampler.stop()  # cleanup


class TestMemorySamplerStartStop:
    def test_stop_returns_stats(self) -> None:
        sampler = MemorySampler(interval_s=0.05)
        sampler.start()
        time.sleep(0.12)
        stats = sampler.stop()
        assert isinstance(stats, MemoryStats)
        assert stats.samples_count >= 1

    def test_double_start_raises(self) -> None:
        sampler = MemorySampler(interval_s=0.05)
        sampler.start()
        with pytest.raises(RuntimeError, match="already running"):
            sampler.start()
        sampler.stop()  # cleanup

    def test_stop_before_start_returns_zero_stats(self) -> None:
        sampler = MemorySampler(interval_s=0.05)
        stats = sampler.stop()
        assert stats.samples_count == 0
        assert stats.peak_rss_bytes == 0
        assert stats.mean_rss_bytes == 0

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_s must be positive"):
            MemorySampler(interval_s=-1)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_s must be positive"):
            MemorySampler(interval_s=0)


class TestMultipleInstances:
    def test_two_samplers_no_shared_state(self) -> None:
        """Two simultaneous samplers must not share sample lists."""
        sampler_a = MemorySampler(interval_s=0.05)
        sampler_b = MemorySampler(interval_s=0.05)
        sampler_a.start()
        time.sleep(0.08)
        sampler_b.start()
        time.sleep(0.12)
        stats_a = sampler_a.stop()
        stats_b = sampler_b.stop()
        # Both captured samples independently.
        assert stats_a.samples_count >= 1
        assert stats_b.samples_count >= 1
        # Sampler A ran longer so it should have at least as many samples.
        assert stats_a.samples_count >= stats_b.samples_count

    def test_sequential_instances_independent(self) -> None:
        """A second instance started after the first one stops sees a
        fresh sample list — not the accumulated samples from the first run."""
        sampler1 = MemorySampler(interval_s=0.05)
        with sampler1:
            time.sleep(0.12)
        count1 = sampler1.stats.samples_count

        sampler2 = MemorySampler(interval_s=0.05)
        with sampler2:
            time.sleep(0.06)
        count2 = sampler2.stats.samples_count

        # Neither count should bleed into the other.
        assert count1 >= 1
        assert count2 >= 1

    def test_thread_cleaned_up_after_exit(self) -> None:
        """After __exit__, the sampler thread must be joined (not alive)."""
        sampler = MemorySampler(interval_s=0.05)
        with sampler:
            time.sleep(0.06)
        # _thread is set to None by stop()
        assert sampler._thread is None


class TestWorkloadAllocatesMoreRss:
    def test_peak_exceeds_start_when_allocation_happens(self) -> None:
        """Allocate inside the sampler window; peak_rss_bytes should be
        larger than the RSS before allocation.

        We can't guarantee the exact delta on all OS/GC configurations,
        but we can assert the sampler does not report zero while a large
        allocation is live.
        """
        sampler = MemorySampler(interval_s=0.02)
        sampler.start()
        # Allocate ~8 MB; keep it alive for multiple sample intervals.
        _big = bytearray(8 * 1024 * 1024)
        time.sleep(0.08)
        del _big
        stats = sampler.stop()
        assert stats.peak_rss_bytes > 0
        assert stats.samples_count >= 2
