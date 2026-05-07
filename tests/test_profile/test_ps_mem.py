"""Tests for PsMemProfiler — daemon-thread-based per-process memory profiling."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vdbbench.profile.ps_mem import PsMemProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_run_result(stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """Return a mock that behaves like subprocess.run()'s CompletedProcess."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = 0
    return result


# ---------------------------------------------------------------------------
# available property
# ---------------------------------------------------------------------------


class TestAvailableProperty:
    def test_available_when_binary_on_path(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value="/usr/bin/ps_mem"):
            profiler = PsMemProfiler(output_dir=tmp_path)
        assert profiler.available is True

    def test_not_available_when_binary_missing(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value=None):
            profiler = PsMemProfiler(output_dir=tmp_path)
        assert profiler.available is False


# ---------------------------------------------------------------------------
# log_path attribute
# ---------------------------------------------------------------------------


class TestLogPath:
    def test_log_path_in_output_dir(self, tmp_path: Path) -> None:
        """log_path must be inside output_dir and named ps_mem.log."""
        with patch("shutil.which", return_value="/usr/bin/ps_mem"):
            profiler = PsMemProfiler(output_dir=tmp_path)
        assert profiler.log_path == tmp_path / "ps_mem.log"
        assert profiler.log_path.parent == tmp_path

    def test_default_interval_is_five(self, tmp_path: Path) -> None:
        """Default interval_s is 5."""
        with patch("shutil.which", return_value="/usr/bin/ps_mem"):
            profiler = PsMemProfiler(output_dir=tmp_path)
        assert profiler._interval_s == 5

    def test_custom_interval_stored(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value="/usr/bin/ps_mem"):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=10)
        assert profiler._interval_s == 10


# ---------------------------------------------------------------------------
# Daemon thread lifecycle
# ---------------------------------------------------------------------------


class TestDaemonThread:
    def test_thread_starts_on_enter(self, tmp_path: Path) -> None:
        """__enter__ must start a daemon thread when ps_mem is available."""
        fake_result = _make_fake_run_result(stdout=b"  10.0 MiB private\n")

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", return_value=fake_result),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=60)
            with profiler:
                assert profiler._thread is not None
                assert profiler._thread.is_alive()
                assert profiler._thread.daemon is True

    def test_thread_exits_on_context_exit(self, tmp_path: Path) -> None:
        """__exit__ must signal the loop and join the thread within 5s."""
        fake_result = _make_fake_run_result(stdout=b"  10.0 MiB private\n")

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", return_value=fake_result),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=60)
            with profiler:
                thread = profiler._thread

        # After the context exits the thread reference is cleared.
        assert profiler._thread is None
        # And the thread itself is no longer alive (joined in __exit__).
        assert thread is not None
        assert not thread.is_alive()

    def test_stop_event_set_on_exit(self, tmp_path: Path) -> None:
        """__exit__ must set the stop event so the loop can exit cleanly."""
        fake_result = _make_fake_run_result(stdout=b"")

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", return_value=fake_result),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=60)
            with profiler:
                pass

        assert profiler._stop_event.is_set()

    def test_output_dir_created_on_enter(self, tmp_path: Path) -> None:
        """__enter__ must create output_dir if it does not already exist."""
        fake_result = _make_fake_run_result(stdout=b"  5.0 MiB\n")
        new_dir = tmp_path / "deep" / "nested" / "ps-mem-out"

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", return_value=fake_result),
        ):
            profiler = PsMemProfiler(output_dir=new_dir, interval_s=60)
            with profiler:
                pass

        assert new_dir.is_dir()

    def test_no_thread_when_binary_missing(self, tmp_path: Path) -> None:
        """When ps_mem is not on PATH, __enter__ must NOT start a thread."""
        with (
            patch("shutil.which", return_value=None),
            pytest.warns(UserWarning, match="ps_mem is not on PATH"),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path)
            with profiler:
                assert profiler._thread is None


# ---------------------------------------------------------------------------
# Soft-fail when ps_mem not on PATH
# ---------------------------------------------------------------------------


class TestPsMemSoftFail:
    def test_no_subprocess_run_when_binary_missing(self, tmp_path: Path) -> None:
        """When ps_mem is not on PATH, subprocess.run must NOT be called."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as mock_run,
            pytest.warns(UserWarning, match="ps_mem is not on PATH"),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path)
            with profiler:
                pass

        mock_run.assert_not_called()

    def test_warns_when_binary_missing(self, tmp_path: Path) -> None:
        """A missing ps_mem binary should emit a UserWarning — not raise."""
        with (
            patch("shutil.which", return_value=None),
            pytest.warns(UserWarning, match="ps_mem is not on PATH"),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path)
            with profiler:
                pass  # must not raise

    def test_no_log_file_when_binary_missing(self, tmp_path: Path) -> None:
        """When ps_mem is missing, the log file must NOT be created."""
        with (
            patch("shutil.which", return_value=None),
            pytest.warns(UserWarning),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path)
            with profiler:
                pass

        assert not profiler.log_path.exists()


# ---------------------------------------------------------------------------
# Output file format — timestamps + append behaviour
# ---------------------------------------------------------------------------


class TestOutputFileFormat:
    def test_log_file_contains_timestamp_header(self, tmp_path: Path) -> None:
        """Each snapshot written to the log must be prefixed with a timestamp banner."""
        # We'll trigger one sample by using a very short interval and allowing
        # the thread to run.  We control subprocess.run to return known output.
        snap_output = b"  Private  +   Shared  =  RAM used\n   8.0 MiB +   2.0 MiB =  10.0 MiB\n"
        fake_result = _make_fake_run_result(stdout=snap_output)

        # Use an Event to allow exactly one sample before we stop.
        sample_taken = threading.Event()

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:  # type: ignore[return]
            result = fake_result
            sample_taken.set()
            return result

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=0)
            # interval_s=0 means the first wait(timeout=0) returns False
            # immediately, so the loop body runs right away.
            with profiler:
                sample_taken.wait(timeout=3)  # give the thread time to write

        assert profiler.log_path.exists()
        content = profiler.log_path.read_text()
        assert "===" in content  # timestamp banner present
        assert "RAM used" in content  # ps_mem output present

    def test_log_is_append_formatted(self, tmp_path: Path) -> None:
        """Each snapshot must be appended (not overwrite) so the log grows."""
        # Pre-seed the log file to confirm append semantics.
        log_path = tmp_path / "ps_mem.log"
        log_path.write_bytes(b"existing-content\n")

        sample_taken = threading.Event()

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:  # type: ignore[return]
            sample_taken.set()
            return _make_fake_run_result(stdout=b"new-sample\n")

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=0)
            with profiler:
                sample_taken.wait(timeout=3)

        content = profiler.log_path.read_text()
        assert "existing-content" in content
        assert "new-sample" in content

    def test_subprocess_run_targets_correct_pid(self, tmp_path: Path) -> None:
        """The sample loop must invoke ps_mem with the current process PID."""
        import os  # noqa: PLC0415

        captured_cmds: list[list[str]] = []
        sample_taken = threading.Event()

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:  # type: ignore[return]
            captured_cmds.append(list(cmd))
            sample_taken.set()
            return _make_fake_run_result(stdout=b"")

        with (
            patch("shutil.which", return_value="/usr/bin/ps_mem"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            profiler = PsMemProfiler(output_dir=tmp_path, interval_s=0)
            with profiler:
                sample_taken.wait(timeout=3)

        assert len(captured_cmds) >= 1
        first_cmd = captured_cmds[0]
        assert first_cmd[0] == "/usr/bin/ps_mem"
        assert "-p" in first_cmd
        pid_idx = first_cmd.index("-p")
        assert first_cmd[pid_idx + 1] == str(os.getpid())


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliPsMemFlag:
    def test_profiler_flag_in_bench_help(self) -> None:
        """``vdbbench bench --help`` must list ``--profiler`` and mention ps_mem."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "--profiler" in result.output
        assert "ps_mem" in result.output

    def test_profiler_ps_mem_in_supported_set(self) -> None:
        """'ps_mem' must be in the _SUPPORTED_PROFILERS frozenset."""
        from vdbbench.cli import _SUPPORTED_PROFILERS  # noqa: PLC0415

        assert "ps_mem" in _SUPPORTED_PROFILERS

    def test_profiler_ps_mem_without_binary_warns(self, tmp_path: Path) -> None:
        """``--profiler ps_mem`` when ps_mem is not installed: soft-fail with
        a warning, not a crash.  We mock the encoded bundle and adapter so
        the bench actually runs."""
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
                    "ps_mem",
                ],
            )
        # Should succeed (exit 0); ps_mem absence is a warning, not fatal.
        assert result.exit_code == 0, f"unexpected exit {result.exit_code}: {result.output}"
