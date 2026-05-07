"""Opt-in ps_mem per-process memory profiler for vdbbench bench runs.

Usage::

    from pathlib import Path
    from vdbbench.profile.ps_mem import PsMemProfiler

    with PsMemProfiler(output_dir=Path("results/profile"), interval_s=5):
        run_bench(...)

``PsMemProfiler`` does **not** spawn a single long-lived subprocess like
``IostatProfiler`` does, because ``ps_mem`` has no continuous-monitoring
mode.  Instead it starts a daemon thread that runs ``ps_mem -p <self_pid>``
repeatedly at ``interval_s`` cadence and appends each snapshot (with a
timestamp header) to ``{output_dir}/ps_mem.log``.

Soft-fail semantics
-------------------
If ``ps_mem`` is not on PATH the profiler emits a ``UserWarning`` and
becomes a no-op context manager — the bench still runs, just without
the detailed memory breakdown.

Requirements
------------
``ps_mem`` is a system binary — it is **not** a Python package dependency
of vdbbench:

* **Linux / Debian/Ubuntu**: ``sudo apt install ps_mem``
* **macOS (Homebrew)**: ``brew install ps_mem``

``ps_mem`` requires root access (or ``sudo``) on many Linux systems to
inspect arbitrary processes.  When run as the current user it can still
profile its own process; elevate with ``sudo`` if you see empty output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import warnings
from pathlib import Path


class PsMemProfiler:
    """Context manager that runs ``ps_mem`` periodically around a bench run.

    Unlike ``PySpyProfiler`` (single long-lived subprocess) and
    ``IostatProfiler`` (single continuous subprocess), ``PsMemProfiler``
    drives a **daemon thread** that re-invokes ``ps_mem -p <pid>`` every
    ``interval_s`` seconds and appends the output to ``ps_mem.log``.
    Each snapshot is prefixed with an ISO-8601 timestamp so the log can
    be correlated with bench phase boundaries.

    Parameters
    ----------
    output_dir:
        Directory where ``ps_mem.log`` will be written.  Created if it
        does not already exist.
    interval_s:
        How often (in seconds) to invoke ``ps_mem``.  Default 5.

    Attributes
    ----------
    log_path:
        Resolved path of the log file.  Available after construction
        (before the context is entered) so callers can log it upfront.
    available:
        ``True`` when the ``ps_mem`` binary is on PATH.  ``False`` when it
        is missing — the context manager is a no-op in that case.
    """

    def __init__(self, output_dir: Path, interval_s: int = 5) -> None:
        self._output_dir = Path(output_dir)
        self._interval_s = interval_s
        self._ps_mem_bin: str | None = shutil.which("ps_mem")
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self.log_path: Path = self._output_dir / "ps_mem.log"

    @property
    def available(self) -> bool:
        """True if the ps_mem binary is on PATH."""
        return self._ps_mem_bin is not None

    # ------------------------------------------------------------------
    # Internal loop — runs on the daemon thread
    # ------------------------------------------------------------------

    def _sample_loop(self, pid: int) -> None:
        """Repeatedly invoke ps_mem and append snapshots to log_path.

        Called on the daemon thread.  Exits when ``_stop_event`` is set.
        Each snapshot is prefixed with a ``=== <ISO-timestamp> ===`` banner
        so multiple samples in the log are easy to navigate.
        """
        assert self._ps_mem_bin is not None  # guaranteed by __enter__ guard
        cmd: list[str] = [self._ps_mem_bin, "-p", str(pid)]
        while not self._stop_event.wait(timeout=self._interval_s):
            try:
                from datetime import UTC, datetime  # noqa: PLC0415

                stamp = datetime.now(UTC).isoformat(timespec="seconds")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,
                )
                with self.log_path.open("ab") as fh:
                    header = f"\n=== {stamp} ===\n".encode()
                    fh.write(header)
                    fh.write(result.stdout)
                    if result.stderr:
                        fh.write(result.stderr)
            except Exception:  # best-effort; must not crash the daemon thread
                # Best-effort: never crash the daemon thread — the bench
                # must keep running even if a single ps_mem invocation fails
                # (e.g. transient permission error).
                pass

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> PsMemProfiler:
        if not self.available:
            warnings.warn(
                "ps_mem is not on PATH — per-process memory profiling skipped. "
                "Install it via: brew install ps_mem (macOS) or "
                "apt install ps_mem (Debian/Ubuntu).",
                stacklevel=2,
            )
            return self
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        pid = os.getpid()
        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(pid,),
            daemon=True,
            name="ps-mem-sampler",
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
