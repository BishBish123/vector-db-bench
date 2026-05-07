"""Background-thread RSS sampler for per-phase memory profiling.

Usage as a context manager::

    with MemorySampler(interval_s=0.1) as sampler:
        do_work()
    stats = sampler.stats   # MemoryStats(peak_rss_bytes, mean_rss_bytes, samples_count)

Usage with explicit start / stop::

    sampler = MemorySampler(interval_s=0.1)
    sampler.start()
    do_work()
    stats = sampler.stop()  # also sets sampler.stats

The sampler polls ``psutil.Process(os.getpid()).memory_info().rss`` at the
configured interval.  When psutil is unavailable every sample returns 0 and
the stats are all-zero — callers should treat ``samples_count == 0`` as
"no data" rather than "zero memory usage".

Multiple ``MemorySampler`` instances are fully independent: each owns its
own ``threading.Thread`` and accumulates its own sample list, so two
simultaneous samplers (e.g. ingest phase and query phase) don't share
state or interfere with each other's results.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryStats:
    """Aggregate RSS statistics captured by a ``MemorySampler`` run.

    Attributes
    ----------
    peak_rss_bytes:
        The maximum RSS observed across all samples (bytes).
    mean_rss_bytes:
        The arithmetic mean of all samples (bytes), or 0 if no samples.
    samples_count:
        How many RSS samples were taken.  A value of 0 means psutil was
        unavailable or the sampler was stopped before any sample fired.
    """

    peak_rss_bytes: int
    mean_rss_bytes: int
    samples_count: int


def _sample_rss() -> int:
    """Return the current process RSS in bytes via psutil.

    Returns 0 if psutil is not installed or the query fails — callers
    treat ``samples_count == 0`` as "no data".
    """
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # pragma: no cover - psutil missing or unreadable
        return 0


class MemorySampler:
    """Background-thread RSS sampler with peak / mean statistics.

    Parameters
    ----------
    interval_s:
        Sampling interval in seconds.  Default 0.1 s (100 ms).

    Thread safety
    -------------
    Each instance owns its own ``threading.Thread``.  Calling ``start()``
    on the same instance twice is a ``RuntimeError``; create a new
    instance for a new phase.  Multiple instances running concurrently
    are safe — they don't share any mutable state.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive; got {interval_s!r}")
        self._interval_s = interval_s
        self._samples: list[int] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats: MemoryStats | None = None

    # ------------------------------------------------------------------
    # Public start / stop API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background sampling.

        Raises ``RuntimeError`` if the sampler is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("MemorySampler is already running; call stop() first")
        self._samples = []
        self._stop_event.clear()
        self._stats = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vdbbench-memory-sampler",
        )
        self._thread.start()

    def stop(self) -> MemoryStats:
        """Stop background sampling and return aggregate statistics.

        Safe to call even if the sampler was never started (returns a
        zero-valued ``MemoryStats``).
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 10 + 1.0)
            self._thread = None
        self._stats = self._compute_stats()
        return self._stats

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    def __enter__(self) -> MemorySampler:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    @property
    def stats(self) -> MemoryStats:
        """Return the most recent stats.

        Raises ``RuntimeError`` if ``stop()`` (or ``__exit__``) has not
        been called yet.
        """
        if self._stats is None:
            raise RuntimeError("MemorySampler has not been stopped yet; call stop() first")
        return self._stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Sampling loop — runs on the background thread."""
        while not self._stop_event.is_set():
            self._samples.append(_sample_rss())
            self._stop_event.wait(timeout=self._interval_s)
        # One final sample at the moment the stop signal fires.
        self._samples.append(_sample_rss())

    def _compute_stats(self) -> MemoryStats:
        samples = self._samples
        if not samples:
            return MemoryStats(peak_rss_bytes=0, mean_rss_bytes=0, samples_count=0)
        peak = max(samples)
        mean = int(sum(samples) / len(samples))
        return MemoryStats(
            peak_rss_bytes=peak,
            mean_rss_bytes=mean,
            samples_count=len(samples),
        )
