"""Opt-in iostat I/O profiler for vdbbench bench runs.

Usage::

    from pathlib import Path
    from vdbbench.profile.iostat import IostatProfiler

    with IostatProfiler(output_dir=Path("results/profile"), interval_s=1):
        run_bench(...)

``IostatProfiler`` spawns ``iostat -x -c -d <interval>`` as a subprocess,
redirects stdout to ``{output_dir}/iostat.txt``, sends SIGINT when the
context exits, and waits for a clean exit.

Soft-fail semantics
-------------------
If ``iostat`` is not on PATH the profiler emits a ``UserWarning`` and
becomes a no-op context manager — the bench still runs, just without
I/O stats.

Requirements
------------
``iostat`` ships with ``sysstat`` on Linux and is part of the base system
on macOS.  It is not available on bare Windows (without WSL2 or Cygwin).
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import warnings
from pathlib import Path


class IostatProfiler:
    """Context manager that records ``iostat`` I/O stats around a bench run.

    Parameters
    ----------
    output_dir:
        Directory where ``iostat.txt`` will be written.  Created if it
        does not already exist.
    interval_s:
        Sampling interval in seconds passed to ``iostat``.  Default 1.

    Attributes
    ----------
    txt_path:
        Resolved path of the text output file.  Available after construction
        (before the context is entered) so callers can log it upfront.
    available:
        ``True`` when the ``iostat`` binary is on PATH.  ``False`` when it
        is missing — the context manager is a no-op in that case.
    """

    def __init__(self, output_dir: Path, interval_s: int = 1) -> None:
        self._output_dir = Path(output_dir)
        self._interval_s = interval_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._iostat_bin: str | None = shutil.which("iostat")
        self.txt_path: Path = self._output_dir / "iostat.txt"

    @property
    def available(self) -> bool:
        """True if the iostat binary is on PATH."""
        return self._iostat_bin is not None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> IostatProfiler:
        if not self.available:
            warnings.warn(
                "iostat is not on PATH — I/O profiling skipped. "
                "Install it via: apt install sysstat (Linux) or it ships "
                "with macOS by default.",
                stacklevel=2,
            )
            return self
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # _iostat_bin is guaranteed non-None here (checked by self.available).
        assert self._iostat_bin is not None
        cmd: list[str] = [
            self._iostat_bin,
            "-x",  # extended stats (disk utilisation, await, etc.)
            "-c",  # CPU stats
            "-d",  # device stats
            str(self._interval_s),
        ]
        fh = self.txt_path.open("wb")
        self._proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.DEVNULL,
        )
        fh.close()  # child holds the fd; parent doesn't need it
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._proc is None:
            return
        try:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            self._proc.kill()
            self._proc.wait()
        finally:
            self._proc = None
