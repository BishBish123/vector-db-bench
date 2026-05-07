"""Opt-in py-spy flame-graph profiler for vdbbench bench runs.

Usage::

    from pathlib import Path
    from vdbbench.profile.pyspy import PySpyProfiler

    with PySpyProfiler(output_dir=Path("results/profile"), sampling_rate_hz=100):
        run_bench(...)

``PySpyProfiler`` spawns ``py-spy record --pid <self> --output <svg> --rate <hz>``
as a subprocess, sends SIGINT when the context exits, and waits for a clean
exit.  The resulting SVG flame graph lands in ``output_dir/profile.svg``.

Soft-fail semantics
-------------------
If ``py-spy`` is not on PATH the profiler logs a warning and becomes a
no-op context manager — the bench still runs, just without a flame graph.
This keeps ``--profile py-spy`` usable in CI where py-spy may not be
installed.

Requirements
------------
``py-spy`` must be installed separately::

    pip install "vdbbench[profile]"   # adds py-spy>=0.4

Linux-only note: ``py-spy record`` may require elevated privileges (``sudo``
or ``SYS_PTRACE`` capability) on Linux kernels with ``ptrace_scope = 1``.
macOS and Docker do not need elevated privileges for self-profiling.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import warnings
from pathlib import Path


class PySpyProfiler:
    """Context manager that records a py-spy SVG flame graph around a bench run.

    Parameters
    ----------
    output_dir:
        Directory where ``profile.svg`` will be written.  Created if it
        does not already exist.
    sampling_rate_hz:
        Samples per second passed to ``py-spy record --rate``.
        Default 100 (py-spy's own default).

    Attributes
    ----------
    svg_path:
        Resolved path of the SVG output file.  Available after construction
        (before the context is entered) so callers can log it upfront.
    available:
        ``True`` when the ``py-spy`` binary is on PATH.  ``False`` when it
        is missing — the context manager is a no-op in that case.
    """

    def __init__(self, output_dir: Path, sampling_rate_hz: int = 100) -> None:
        self._output_dir = Path(output_dir)
        self._rate = sampling_rate_hz
        self._proc: subprocess.Popen[bytes] | None = None
        self._pyspy_bin: str | None = shutil.which("py-spy")
        self.svg_path: Path = self._output_dir / "profile.svg"

    @property
    def available(self) -> bool:
        """True if the py-spy binary is on PATH."""
        return self._pyspy_bin is not None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> PySpyProfiler:
        if not self.available:
            warnings.warn(
                "py-spy is not on PATH — profiling skipped. "
                "Install it with: pip install 'vdbbench[profile]'",
                stacklevel=2,
            )
            return self
        self._output_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        # _pyspy_bin is guaranteed non-None here (checked by self.available guard above).
        assert self._pyspy_bin is not None
        cmd: list[str] = [
            self._pyspy_bin,
            "record",
            "--pid",
            str(pid),
            "--output",
            str(self.svg_path),
            "--rate",
            str(self._rate),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
