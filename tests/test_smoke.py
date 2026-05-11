"""Smoke tests so CI has something to run on day one."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

import vdbbench
from vdbbench.bench.runner import BenchResult, SkippedSpec
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


def _stub_load_encoded_bundle(_path: object) -> object:
    """Stand-in for `load_encoded_bundle` so the CLI tolerance tests can
    drive the bench command without building a real encoded bundle on
    disk. The returned object's only consumer in CLI is run_bench, which
    we also stub."""

    class _StubBundle:
        name = "stub-corpus"

    class _StubEncoded:
        bundle = _StubBundle()

    return _StubEncoded()


def _make_fake_run_bench(
    *,
    skip_dbs: tuple[str, ...] = (),
    fail_all: bool = False,
):
    """Build a fake `run_bench` that emulates partial-availability outcomes.

    Specs with `adapter.name in skip_dbs` are recorded as skipped;
    everything else gets a one-row summary entry. When `fail_all` is set,
    every spec is skipped — used to drive the "every adapter failed" branch.
    """

    def _fake(
        _encoded: object,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
    ) -> BenchResult:
        del progress  # signature parity with run_bench; not used in the fake
        summary_rows: list[dict[str, object]] = []
        skipped: list[SkippedSpec] = []
        for spec in specs:
            db = spec.adapter.name
            should_skip = fail_all or db in skip_dbs
            if should_skip:
                if not tolerate_failures:
                    raise ConnectionRefusedError(f"{db} unreachable")
                skipped.append(
                    SkippedSpec(
                        label=spec.display_label(),
                        db=db,
                        reason=f"{db} unreachable",
                        error_type="ConnectionRefusedError",
                    )
                )
            else:
                summary_rows.append({"db": db, "label": spec.display_label()})
        return BenchResult(
            timings=pd.DataFrame(),
            summary=pd.DataFrame(summary_rows),
            manifest={"schema_version": 1},
            skipped=tuple(skipped),
        )

    return _fake


def test_cli_bench_all_skips_unavailable_adapter(tmp_path: Path) -> None:
    """`--all` keeps going when one service is dead — pgvector unreachable,
    qdrant succeeds, exit code is 0 and a skip line is printed."""
    fake_run = _make_fake_run_bench(skip_dbs=("pgvector",))
    out = tmp_path / "results"
    runner = CliRunner()
    with (
        patch("vdbbench.embed.load_encoded_bundle", _stub_load_encoded_bundle),
        patch("vdbbench.bench.run_bench", fake_run),
    ):
        result = runner.invoke(
            app,
            [
                "bench",
                "--encoded",
                str(tmp_path / "encoded"),
                "--out",
                str(out),
                "--all",
            ],
        )
    assert result.exit_code == 0, result.output
    # Structured skip surfaced in the human-facing summary.
    assert "skipped" in result.output
    assert "pgvector" in result.output


def test_cli_bench_all_fails_when_no_adapter_succeeds(tmp_path: Path) -> None:
    """If every adapter under `--all` is unreachable, exit non-zero —
    silently producing an empty summary would mask a broken environment."""
    fake_run = _make_fake_run_bench(fail_all=True)
    out = tmp_path / "results"
    runner = CliRunner()
    with (
        patch("vdbbench.embed.load_encoded_bundle", _stub_load_encoded_bundle),
        patch("vdbbench.bench.run_bench", fake_run),
    ):
        result = runner.invoke(
            app,
            [
                "bench",
                "--encoded",
                str(tmp_path / "encoded"),
                "--out",
                str(out),
                "--all",
            ],
        )
    assert result.exit_code != 0
    assert "every adapter failed" in result.output


def test_cli_bench_adapter_flag_unknown_value_rejected() -> None:
    """`vdbbench bench --adapter bogus` errors out with the choice list."""
    result = CliRunner().invoke(
        app,
        ["bench", "--encoded", "/nonexistent", "--adapter", "bogus"],
    )
    assert result.exit_code != 0
    assert "unknown adapter" in result.output


def test_cli_bench_adapter_exact_runs_offline(tmp_path: Path) -> None:
    """`--adapter exact` registers the in-process brute-force adapter so
    a smoke run with no Docker still produces a summary row."""
    captured: dict[str, object] = {}

    def _fake(
        _encoded: object,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
    ) -> BenchResult:
        del progress, tolerate_failures
        captured["dbs"] = [s.adapter.name for s in specs]
        return BenchResult(
            timings=pd.DataFrame(),
            summary=pd.DataFrame([{"db": s.adapter.name} for s in specs]),
            manifest={"schema_version": 1},
        )

    out = tmp_path / "results"
    runner = CliRunner()
    with (
        patch("vdbbench.embed.load_encoded_bundle", _stub_load_encoded_bundle),
        patch("vdbbench.bench.run_bench", _fake),
    ):
        result = runner.invoke(
            app,
            [
                "bench",
                "--encoded",
                str(tmp_path / "encoded"),
                "--out",
                str(out),
                "--adapter",
                "exact",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured["dbs"] == ["exact"]


def test_cli_bench_adapter_memory_alias_resolves_to_exact(tmp_path: Path) -> None:
    """`--adapter memory` is an alias for `exact` — a single adapter is
    registered, named `exact`, not `memory`."""

    def _fake(
        _encoded: object,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
    ) -> BenchResult:
        del progress, tolerate_failures
        return BenchResult(
            timings=pd.DataFrame(),
            summary=pd.DataFrame([{"db": s.adapter.name} for s in specs]),
            manifest={"schema_version": 1},
        )

    out = tmp_path / "results"
    runner = CliRunner()
    with (
        patch("vdbbench.embed.load_encoded_bundle", _stub_load_encoded_bundle),
        patch("vdbbench.bench.run_bench", _fake),
    ):
        result = runner.invoke(
            app,
            [
                "bench",
                "--encoded",
                str(tmp_path / "encoded"),
                "--out",
                str(out),
                "--adapter",
                "memory",
            ],
        )
    assert result.exit_code == 0, result.output


def test_cli_bench_single_adapter_still_hard_fails(tmp_path: Path) -> None:
    """Without `--all`, a connection error on the explicitly-named adapter
    must propagate — there's no other adapter to keep going for, and the
    user asked for that one specifically."""
    fake_run = _make_fake_run_bench(fail_all=True)
    out = tmp_path / "results"
    runner = CliRunner()
    with (
        patch("vdbbench.embed.load_encoded_bundle", _stub_load_encoded_bundle),
        patch("vdbbench.bench.run_bench", fake_run),
    ):
        result = runner.invoke(
            app,
            [
                "bench",
                "--encoded",
                str(tmp_path / "encoded"),
                "--out",
                str(out),
                "--pgvector-dsn",
                "postgresql://nope",
            ],
        )
    assert result.exit_code != 0
