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
