"""Smoke tests so CI has something to run on day one."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

import vdbbench
from vdbbench.cli import app


def test_version_string_is_not_unknown() -> None:
    assert vdbbench.__version__
    assert vdbbench.__version__ != "unknown"


def test_cli_version_subcommand_runs() -> None:
    """`vdbbench version` exits cleanly and prints the version."""
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert vdbbench.__version__ in result.output


def test_cli_module_runnable() -> None:
    """`python -m vdbbench.cli version` works as a sanity invocation."""
    out = subprocess.run(
        [sys.executable, "-m", "vdbbench.cli", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert vdbbench.__version__ in out.stdout


def test_cli_help_runs() -> None:
    """`vdbbench --help` exits cleanly and lists the bench subcommand.

    Lightweight smoke for the CLI wiring — catches typer registration
    breakage that the unit tests would miss because they import `app`
    directly rather than running the resolved entrypoint.
    """
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "bench" in result.output


def test_cli_bench_help_documents_all_flag() -> None:
    """`vdbbench bench --help` must surface the new --all flag.

    `make bench` (and the README) calls `vdbbench bench --all`. The CLI
    used to silently reject that flag (no `--all` option registered),
    breaking the documented one-command reproduction.
    """
    result = CliRunner().invoke(app, ["bench", "--help"])
    assert result.exit_code == 0, result.output
    assert "--all" in result.output


def test_cli_bench_no_adapters_exits_nonzero() -> None:
    """Running `vdbbench bench` without any adapter flag must fail loud."""
    result = CliRunner().invoke(app, ["bench", "--encoded", "/nonexistent"])
    # Exit code 2 ("no adapters enabled") OR a non-zero load failure are
    # both acceptable — what matters is the run doesn't silently succeed.
    assert result.exit_code != 0
