"""Tests for PySpyProfiler — subprocess-based flame-graph integration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vdbbench.profile.pyspy import PySpyProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_proc(returncode: int = 0) -> MagicMock:
    """Return a mock that behaves like a subprocess.Popen object."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.returncode = returncode
    proc.wait.return_value = returncode
    proc.send_signal.return_value = None
    proc.kill.return_value = None
    return proc


# ---------------------------------------------------------------------------
# available property
# ---------------------------------------------------------------------------


class TestAvailableProperty:
    def test_available_when_binary_on_path(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/py-spy"):
            profiler = PySpyProfiler(output_dir=tmp_path)
        assert profiler.available is True

    def test_not_available_when_binary_missing(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value=None):
            profiler = PySpyProfiler(output_dir=tmp_path)
        assert profiler.available is False


# ---------------------------------------------------------------------------
# Subprocess spawn smoke
# ---------------------------------------------------------------------------


class TestPySpyProfilerSubprocessArgs:
    def test_spawns_correct_command(self, tmp_path: Path) -> None:
        """__enter__ must call Popen with the right py-spy arguments."""
        fake_proc = _make_fake_proc()

        with (
            patch("shutil.which", return_value="/usr/bin/py-spy"),
            patch("subprocess.Popen", return_value=fake_proc) as mock_popen,
        ):
            profiler = PySpyProfiler(output_dir=tmp_path, sampling_rate_hz=50)
            with profiler:
                pass

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]

        assert cmd[0] == "/usr/bin/py-spy"
        assert "record" in cmd
        assert "--pid" in cmd
        assert str(os.getpid()) in cmd  # must target the current process
        assert "--output" in cmd
        output_idx = cmd.index("--output")
        assert "profile.svg" in cmd[output_idx + 1]
        assert "--rate" in cmd
        rate_idx = cmd.index("--rate")
        assert cmd[rate_idx + 1] == "50"

    def test_sends_sigint_on_exit(self, tmp_path: Path) -> None:
        """__exit__ must SIGINT the subprocess so py-spy flushes the SVG."""
        import signal  # noqa: PLC0415

        fake_proc = _make_fake_proc()

        with (
            patch("shutil.which", return_value="/usr/bin/py-spy"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = PySpyProfiler(output_dir=tmp_path)
            with profiler:
                pass

        fake_proc.send_signal.assert_called_once_with(signal.SIGINT)
        fake_proc.wait.assert_called()

    def test_svg_path_in_output_dir(self, tmp_path: Path) -> None:
        """svg_path must be inside output_dir and named profile.svg."""
        with patch("shutil.which", return_value="/usr/bin/py-spy"):
            profiler = PySpyProfiler(output_dir=tmp_path)
        assert profiler.svg_path == tmp_path / "profile.svg"
        assert profiler.svg_path.parent == tmp_path

    def test_output_dir_created_on_enter(self, tmp_path: Path) -> None:
        """__enter__ must create output_dir if it does not already exist."""
        fake_proc = _make_fake_proc()
        new_dir = tmp_path / "deep" / "nested" / "profile"

        with (
            patch("shutil.which", return_value="/usr/bin/py-spy"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = PySpyProfiler(output_dir=new_dir)
            with profiler:
                pass

        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# Soft-fail when py-spy not on PATH
# ---------------------------------------------------------------------------


class TestPySpySoftFail:
    def test_no_popen_when_binary_missing(self, tmp_path: Path) -> None:
        """When py-spy is not on PATH, __enter__ must NOT call Popen."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.Popen") as mock_popen,
            pytest.warns(UserWarning, match="py-spy is not on PATH"),
        ):
            profiler = PySpyProfiler(output_dir=tmp_path)
            with profiler:
                pass

        mock_popen.assert_not_called()

    def test_warns_when_binary_missing(self, tmp_path: Path) -> None:
        """A missing py-spy binary should emit a UserWarning — not raise."""
        with (
            patch("shutil.which", return_value=None),
            pytest.warns(UserWarning, match="py-spy is not on PATH"),
        ):
            profiler = PySpyProfiler(output_dir=tmp_path)
            with profiler:
                pass  # must not raise


# ---------------------------------------------------------------------------
# Timeout / kill fallback on __exit__
# ---------------------------------------------------------------------------


class TestPySpyExitFallback:
    def test_kills_subprocess_on_timeout(self, tmp_path: Path) -> None:
        """If py-spy doesn't exit within the timeout, kill() is called."""
        import signal  # noqa: PLC0415

        fake_proc = MagicMock(spec=subprocess.Popen)
        fake_proc.send_signal.return_value = None
        fake_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="py-spy", timeout=15)
        fake_proc.kill.return_value = None
        # Second wait (after kill) must succeed.
        fake_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="py-spy", timeout=15), None]

        with (
            patch("shutil.which", return_value="/usr/bin/py-spy"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = PySpyProfiler(output_dir=tmp_path)
            with profiler:
                pass

        fake_proc.send_signal.assert_called_with(signal.SIGINT)
        fake_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# CLI integration: --profiler flag visible in help
# ---------------------------------------------------------------------------


class TestCliProfilerFlag:
    def test_profiler_flag_in_bench_help(self) -> None:
        """``vdbbench bench --help`` must list ``--profiler`` so users know
        the flag exists without reading source."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "--profiler" in result.output

    def test_unknown_profiler_raises_value_error(self) -> None:
        """Passing an unsupported profiler name must produce a clean error."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bench",
                "--adapter",
                "exact",
                "--encoded",
                "data/encoded",  # doesn't need to exist for this error check
                "--profiler",
                "gprof",  # not a supported profiler
            ],
        )
        # Should fail with exit code 2 (user-facing error), not 0.
        assert result.exit_code != 0

    def test_profiler_py_spy_without_binary_warns(self, tmp_path: Path) -> None:
        """``--profiler py-spy`` when py-spy is not installed: soft-fail with
        a warning, not a crash.  We mock the encoded bundle and adapter so
        the bench actually runs."""
        import warnings  # noqa: PLC0415

        import pandas as pd  # noqa: PLC0415
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415
        from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415
        from vdbbench.embed.encoder import (  # noqa: PLC0415
            FakeEncoder,
            encode_corpus,
        )

        bundle = CorpusBundle(
            name="tiny",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["a", "b"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["a"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1.0]}),
        )
        encoded = encode_corpus(bundle, FakeEncoder(dim=4))
        encoded.save(tmp_path / "encoded")

        runner = CliRunner()
        # Downgrade UserWarning to "always" so pytest's filterwarnings="error"
        # doesn't convert the soft-fail py-spy warning into an exception inside
        # the CliRunner invocation — we want to verify the bench still exits 0.
        with (
            patch("shutil.which", return_value=None),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("always", UserWarning)
            result = runner.invoke(
                app,
                [
                    "bench",
                    "--encoded",
                    str(tmp_path / "encoded"),
                    "--out",
                    str(tmp_path / "out"),
                    "--adapter",
                    "exact",
                    "--profiler",
                    "py-spy",
                ],
            )
        # Should succeed (exit 0); py-spy absence is a warning, not a fatal error.
        assert result.exit_code == 0, f"unexpected exit {result.exit_code}: {result.output}"
