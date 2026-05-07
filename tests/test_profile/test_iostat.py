"""Tests for IostatProfiler — subprocess-based I/O stats integration."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vdbbench.profile.iostat import IostatProfiler

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
        with patch("shutil.which", return_value="/usr/bin/iostat"):
            profiler = IostatProfiler(output_dir=tmp_path)
        assert profiler.available is True

    def test_not_available_when_binary_missing(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value=None):
            profiler = IostatProfiler(output_dir=tmp_path)
        assert profiler.available is False


# ---------------------------------------------------------------------------
# Subprocess spawn
# ---------------------------------------------------------------------------


class TestIostatProfilerSubprocessArgs:
    def test_spawns_correct_command(self, tmp_path: Path) -> None:
        """__enter__ must call Popen with the expected iostat arguments."""
        fake_proc = _make_fake_proc()

        with (
            patch("shutil.which", return_value="/usr/bin/iostat"),
            patch("subprocess.Popen", return_value=fake_proc) as mock_popen,
        ):
            profiler = IostatProfiler(output_dir=tmp_path, interval_s=2)
            with profiler:
                pass

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]

        assert cmd[0] == "/usr/bin/iostat"
        assert "-x" in cmd
        assert "-c" in cmd
        assert "-d" in cmd
        assert "2" in cmd  # interval_s

    def test_sends_sigint_on_exit(self, tmp_path: Path) -> None:
        """__exit__ must SIGINT the subprocess so iostat flushes its buffer."""
        fake_proc = _make_fake_proc()

        with (
            patch("shutil.which", return_value="/usr/bin/iostat"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = IostatProfiler(output_dir=tmp_path)
            with profiler:
                pass

        fake_proc.send_signal.assert_called_once_with(signal.SIGINT)
        fake_proc.wait.assert_called()

    def test_txt_path_in_output_dir(self, tmp_path: Path) -> None:
        """txt_path must be inside output_dir and named iostat.txt."""
        with patch("shutil.which", return_value="/usr/bin/iostat"):
            profiler = IostatProfiler(output_dir=tmp_path)
        assert profiler.txt_path == tmp_path / "iostat.txt"
        assert profiler.txt_path.parent == tmp_path

    def test_output_dir_created_on_enter(self, tmp_path: Path) -> None:
        """__enter__ must create output_dir if it does not already exist."""
        fake_proc = _make_fake_proc()
        new_dir = tmp_path / "deep" / "nested" / "iostat-out"

        with (
            patch("shutil.which", return_value="/usr/bin/iostat"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = IostatProfiler(output_dir=new_dir)
            with profiler:
                pass

        assert new_dir.is_dir()

    def test_default_interval_is_one(self, tmp_path: Path) -> None:
        """Default interval_s is 1."""
        with patch("shutil.which", return_value="/usr/bin/iostat"):
            profiler = IostatProfiler(output_dir=tmp_path)
        assert profiler._interval_s == 1


# ---------------------------------------------------------------------------
# Soft-fail when iostat not on PATH
# ---------------------------------------------------------------------------


class TestIostatSoftFail:
    def test_no_popen_when_binary_missing(self, tmp_path: Path) -> None:
        """When iostat is not on PATH, __enter__ must NOT call Popen."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.Popen") as mock_popen,
            pytest.warns(UserWarning, match="iostat is not on PATH"),
        ):
            profiler = IostatProfiler(output_dir=tmp_path)
            with profiler:
                pass

        mock_popen.assert_not_called()

    def test_warns_when_binary_missing(self, tmp_path: Path) -> None:
        """A missing iostat binary should emit a UserWarning — not raise."""
        with (
            patch("shutil.which", return_value=None),
            pytest.warns(UserWarning, match="iostat is not on PATH"),
        ):
            profiler = IostatProfiler(output_dir=tmp_path)
            with profiler:
                pass  # must not raise


# ---------------------------------------------------------------------------
# Timeout / kill fallback on __exit__
# ---------------------------------------------------------------------------


class TestIostatExitFallback:
    def test_kills_subprocess_on_timeout(self, tmp_path: Path) -> None:
        """If iostat doesn't exit within the timeout, kill() is called."""
        fake_proc = MagicMock(spec=subprocess.Popen)
        fake_proc.send_signal.return_value = None
        fake_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="iostat", timeout=15),
            None,
        ]
        fake_proc.kill.return_value = None

        with (
            patch("shutil.which", return_value="/usr/bin/iostat"),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            profiler = IostatProfiler(output_dir=tmp_path)
            with profiler:
                pass

        fake_proc.send_signal.assert_called_with(signal.SIGINT)
        fake_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# CLI integration: --profiler iostat flag visible in help and validation
# ---------------------------------------------------------------------------


class TestCliIostatFlag:
    def test_profiler_flag_in_bench_help(self) -> None:
        """``vdbbench bench --help`` must list ``--profiler`` so users know
        the flag exists without reading source."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "--profiler" in result.output
        # Both profilers should be described.
        assert "iostat" in result.output

    def test_profiler_iostat_in_supported_set(self) -> None:
        """'iostat' must be in the _SUPPORTED_PROFILERS frozenset."""
        from vdbbench.cli import _SUPPORTED_PROFILERS  # noqa: PLC0415

        assert "iostat" in _SUPPORTED_PROFILERS

    def test_profiler_iostat_without_binary_warns(self, tmp_path: Path) -> None:
        """``--profiler iostat`` when iostat is not installed: soft-fail with
        a warning, not a crash."""
        import warnings  # noqa: PLC0415

        import pandas as pd  # noqa: PLC0415
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415
        from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415
        from vdbbench.embed.encoder import FakeEncoder, encode_corpus  # noqa: PLC0415

        bundle = CorpusBundle(
            name="tiny",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["a", "b"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["a"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1.0]}),
        )
        encoded = encode_corpus(bundle, FakeEncoder(dim=4))
        encoded.save(tmp_path / "encoded")

        runner = CliRunner()
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
                    "iostat",
                ],
            )
        # Should succeed (exit 0); iostat absence is a warning, not fatal.
        assert result.exit_code == 0, f"unexpected exit {result.exit_code}: {result.output}"
