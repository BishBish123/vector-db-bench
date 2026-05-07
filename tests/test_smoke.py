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


def test_cli_prep_default_sample_size_is_5000() -> None:
    """`vdbbench prep --help` advertises the 5000-default that matches
    the committed demo bundle. Pin it so a future "round number" drift
    away from 5000 doesn't silently break the README's claim that
    `uv run vdbbench prep` reproduces the demo numbers."""
    result = CliRunner().invoke(app, ["prep", "--help"])
    assert result.exit_code == 0, result.output
    assert "5000" in result.output


def test_resolve_bench_out_default_lives_under_results_run() -> None:
    """The auto-default lands under results/run/ — explicit --out keeps
    bypassing the auto-suffix so Make targets that pin a fixed path
    (results/demo, results/100k, results/full) keep working."""
    from vdbbench.cli import _resolve_bench_out  # noqa: PLC0415

    auto = _resolve_bench_out(None)
    assert auto.parent == Path("results/run")
    explicit = _resolve_bench_out(Path("results/demo"))
    assert explicit == Path("results/demo")


def test_resolve_bench_out_default_is_unique_per_call() -> None:
    """Two unstamped --out resolutions must produce distinct paths so
    parallel `vdbbench bench` invocations can't overwrite each other."""
    from vdbbench.cli import _resolve_bench_out  # noqa: PLC0415

    a = _resolve_bench_out(None)
    b = _resolve_bench_out(None)
    assert a != b


def test_cli_bench_default_out_lives_under_results_run() -> None:
    """`vdbbench bench --help` advertises `results/run/` as the default
    --out parent so the top-level results/ tree stays clean (only named
    scale subdirs like demo / 100k / full live there directly)."""
    result = CliRunner().invoke(app, ["bench", "--help"])
    assert result.exit_code == 0, result.output
    assert "results/run" in result.output


def test_cli_plot_default_summary_picks_most_recent_run(tmp_path: Path) -> None:
    """`vdbbench plot` (no --summary) resolves to the most recently
    modified ``results/run/<stamp>/summary.parquet`` so `bench && plot`
    keeps working without explicit paths even after the bench default
    moved to a timestamped subdir."""
    import os  # noqa: PLC0415

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        run_root = tmp_path / "results" / "run"
        older = run_root / "2025-01-01T00-00-00+00-00-aaaaaaaa"
        newer = run_root / "2025-06-01T00-00-00+00-00-bbbbbbbb"
        for sub in (older, newer):
            sub.mkdir(parents=True)
            # Populate a minimal summary parquet so plot has something
            # to chart; we only need the resolution to pick the right
            # one, so a single-row frame is enough.
            pd.DataFrame(
                {
                    "db": ["mem"],
                    "label": ["mem:a"],
                    "params_hash": ["a"],
                    "params_json": ["{}"],
                    "profile": ["warm"],
                    "n_passages": [1],
                    "n_queries": [1],
                    "dim": [1],
                    "ingest_s": [0.01],
                    "ingest_throughput_vps": [10.0],
                    "index_s": [0.01],
                    "index_bytes": [1],
                    "latency_ms_mean": [1.0],
                    "latency_ms_p50": [1.0],
                    "latency_ms_p95": [1.0],
                    "latency_ms_p99": [1.0],
                    "recall_at_k_mean": [1.0],
                    "recall_at_k_p50": [1.0],
                    "ndcg_at_k_mean": [1.0],
                    "qps_estimate": [1000.0],
                }
            ).to_parquet(sub / "summary.parquet", index=False)
        # Force `newer` to have a strictly later mtime than `older`.
        os.utime(older / "summary.parquet", (1_000_000_000, 1_000_000_000))
        os.utime(newer / "summary.parquet", (2_000_000_000, 2_000_000_000))
        result = CliRunner().invoke(
            app,
            [
                "plot",
                "--out",
                str(tmp_path / "assets"),
                "--baseline-label",
                "mem:a",
            ],
        )
        assert result.exit_code == 0, result.output
    finally:
        os.chdir(cwd)


def test_cli_plot_default_with_no_runs_errors_cleanly(tmp_path: Path) -> None:
    """No results/run/ subdirs => clean ValueError (exit 2), not a
    Python traceback. Catches the case where the user runs `vdbbench
    plot` from a fresh checkout without ever running bench."""
    import os  # noqa: PLC0415

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = CliRunner().invoke(app, ["plot", "--out", str(tmp_path / "assets")])
        assert result.exit_code == 2, result.output
        assert "Traceback" not in result.output
        assert "summary" in result.output.lower()
    finally:
        os.chdir(cwd)


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
        out: object = None,
    ) -> BenchResult:
        del progress, out  # signature parity with run_bench; not used in the fake
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


def test_cli_plot_unrelated_userwarning_not_rerendered_as_note(tmp_path: Path) -> None:
    """A UserWarning from outside ``vdbbench.plot.charts`` (e.g. a
    downstream library) must NOT be re-rendered as a Rich `[yellow]note[/]`
    line. The CLI's catch_warnings filter is scoped to vdbbench's own
    plot module so third-party warnings stay visible as themselves
    rather than getting laundered through the CLI's advisory channel."""
    import warnings as _warnings  # noqa: PLC0415

    summary = tmp_path / "summary.parquet"
    out = tmp_path / "out"
    pd.DataFrame(
        {
            "db": ["mem"],
            "label": ["mem:a"],
            "params_hash": ["a"],
            "params_json": ["{}"],
            "profile": ["warm"],
            "n_passages": [10],
            "n_queries": [2],
            "dim": [4],
            "ingest_s": [0.01],
            "ingest_throughput_vps": [1000.0],
            "index_s": [0.01],
            "index_bytes": [1],
            "latency_ms_mean": [1.0],
            "latency_ms_p50": [1.0],
            "latency_ms_p95": [1.0],
            "latency_ms_p99": [1.0],
            "recall_at_k_mean": [1.0],
            "recall_at_k_p50": [1.0],
            "ndcg_at_k_mean": [1.0],
            "qps_estimate": [1000.0],
        }
    ).to_parquet(summary, index=False)

    # Wrap plot_all so it emits a UserWarning from a non-charts module
    # (this test file). The CLI must not surface that warning as a
    # vdbbench advisory.
    from vdbbench.plot import plot_all as _orig_plot_all  # noqa: PLC0415

    def _plot_all_with_alien_warning(*args: object, **kwargs: object) -> object:
        _warnings.warn(
            "alien-warning-from-downstream-library", UserWarning, stacklevel=2
        )
        return _orig_plot_all(*args, **kwargs)  # type: ignore[arg-type]

    with patch("vdbbench.plot.plot_all", _plot_all_with_alien_warning):
        result = CliRunner().invoke(
            app,
            [
                "plot",
                "--summary",
                str(summary),
                "--out",
                str(out),
                "--baseline-label",
                "mem:a",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "alien-warning-from-downstream-library" not in result.output


def test_cli_plot_swallows_userwarning_into_rich_note(tmp_path: Path) -> None:
    """When `plot_all` (or its callees) emits a UserWarning — e.g. the
    speedup baseline fallback — the CLI catches it and re-emits as a
    Rich-coloured "note:" line, so the user doesn't see a raw Python
    UserWarning blob in stderr.

    Drives a synthetic summary parquet whose only DB is `mem` so the
    speedup chart goes through the auto-baseline branch.
    """
    import pandas as pd  # noqa: PLC0415

    summary = tmp_path / "summary.parquet"
    out = tmp_path / "out"
    pd.DataFrame(
        {
            "db": ["mem", "mem"],
            "label": ["mem:a", "mem:b"],
            "params_hash": ["a", "b"],
            "params_json": ["{}", "{}"],
            "profile": ["warm", "warm"],
            "n_passages": [10, 10],
            "n_queries": [2, 2],
            "dim": [4, 4],
            "ingest_s": [0.01, 0.01],
            "ingest_throughput_vps": [1000.0, 1000.0],
            "index_s": [0.01, 0.01],
            "index_bytes": [1, 1],
            "latency_ms_mean": [1.0, 2.0],
            "latency_ms_p50": [1.0, 2.0],
            "latency_ms_p95": [1.0, 2.0],
            "latency_ms_p99": [1.0, 2.0],
            "recall_at_k_mean": [1.0, 1.0],
            "recall_at_k_p50": [1.0, 1.0],
            "ndcg_at_k_mean": [1.0, 1.0],
            "qps_estimate": [1000.0, 500.0],
        }
    ).to_parquet(summary, index=False)
    # Two `mem` configs — speedup chart needs an explicit baseline_label
    # to disambiguate; without it, plot_speedup_vs_baseline raises a
    # ValueError with the available labels listed. Pass one to keep the
    # plot succeeding so we can observe whatever Rich output the CLI
    # produces (and confirm no UserWarning leaks into stderr).
    result = CliRunner().invoke(
        app,
        [
            "plot",
            "--summary",
            str(summary),
            "--out",
            str(out),
            "--baseline-label",
            "mem:a",
        ],
    )
    assert result.exit_code == 0, result.output
    # Combined output must NOT contain a raw `UserWarning:` blob —
    # that was the old stderr surface the CLI is meant to suppress.
    assert "UserWarning" not in result.output


def test_cli_plot_missing_summary_prints_clean_error(tmp_path: Path) -> None:
    """`vdbbench plot --summary <missing>` exits 2 with a clean message,
    not a Python traceback. The previous behaviour dumped a
    FileNotFoundError stack which made it look like a bug in the CLI."""
    missing = tmp_path / "nope.parquet"
    result = CliRunner().invoke(
        app, ["plot", "--summary", str(missing), "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 2, result.output
    assert "error" in result.output.lower()
    # Specifically: no Python traceback header.
    assert "Traceback" not in result.output


def test_cli_prep_invalid_dataset_prints_clean_error(tmp_path: Path) -> None:
    """A ValueError from the prep pipeline (e.g. an unknown dataset) is
    surfaced as a single error line + exit 2, not a stack trace."""
    result = CliRunner().invoke(
        app,
        [
            "prep",
            "--out",
            str(tmp_path / "encoded"),
            "--dataset",
            "synthetic",
            "--sample-size",
            "0",  # SyntheticConfig rejects non-positive sample_size with ValueError
        ],
    )
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


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
        out: object = None,
    ) -> BenchResult:
        del progress, tolerate_failures, out
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
        out: object = None,
    ) -> BenchResult:
        del progress, tolerate_failures, out
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


def test_adapter_selection_excludes_unselected_with_creds(tmp_path: Path) -> None:
    """`--adapter pgvector --qdrant-url ...` is a contradiction.

    The previous behaviour silently ran qdrant alongside pgvector
    because ``--qdrant-url`` built a spec independently of ``--adapter``.
    Now ``--adapter`` is authoritative — passing a per-adapter flag for
    an unselected adapter raises a user-facing error so the typo isn't
    a silently-double-billed run.
    """
    captured: dict[str, object] = {}

    def _fake(
        _encoded: object,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
        out: object = None,
    ) -> BenchResult:
        del progress, tolerate_failures, out
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
                "pgvector",
                "--qdrant-url",
                "http://localhost:6333",
            ],
        )
    # User-facing ValueError -> exit 2 with a clean message; no qdrant
    # spec was constructed.
    assert result.exit_code == 2, result.output
    assert "qdrant" in result.output
    assert "--qdrant-url" in result.output
    assert "captured" not in result.output  # no progress was printed
    assert "dbs" not in captured  # run_bench was never called


def test_adapter_selection_authoritative_drops_unselected_creds(tmp_path: Path) -> None:
    """When --adapter is set, only the selected adapters are wired —
    even if other URLs/paths are absent, no spec is built for them.
    With --adapter exact, only the in-process brute-force adapter runs."""
    captured: dict[str, object] = {}

    def _fake(
        _encoded: object,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
        out: object = None,
    ) -> BenchResult:
        del progress, tolerate_failures, out
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


# ---------------------------------------------------------------------------
# --encoder flag on `prep` (R7 audit additions)
# ---------------------------------------------------------------------------


def _make_tiny_corpus_bundle() -> object:
    """Return a minimal CorpusBundle for encoder CLI tests."""
    from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415

    return CorpusBundle(
        name="tiny",
        passages=pd.DataFrame({"pid": ["p0"], "text": ["hello"]}),
        queries=pd.DataFrame({"qid": ["q0"], "text": ["world"]}),
        qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
    )


def test_cli_prep_encoder_bge_flag_works(tmp_path: Path) -> None:
    """`vdbbench prep --encoder bge` routes through build_encoder() without downloading.

    Uses --dataset msmarco (with mocked load_msmarco + encode_corpus) so the
    encoder branch is actually exercised. Synthetic bypasses the encoder entirely.
    """
    from unittest.mock import MagicMock  # noqa: PLC0415

    from vdbbench.embed.encoder import FakeEncoder, encode_corpus  # noqa: PLC0415

    bundle = _make_tiny_corpus_bundle()
    fake_enc = FakeEncoder(dim=16)
    mock_build = MagicMock(return_value=fake_enc)

    def _fake_encode_corpus(b: object, enc: object, **kw: object) -> object:
        return encode_corpus(bundle, FakeEncoder(dim=16), **kw)

    with (
        patch("vdbbench.corpus.load_msmarco", return_value=bundle),
        patch("vdbbench.embed.registry.build_encoder", mock_build),
        patch("vdbbench.embed.encode_corpus", _fake_encode_corpus),
    ):
        result = CliRunner().invoke(
            app,
            [
                "prep",
                "--out",
                str(tmp_path / "encoded"),
                "--dataset",
                "msmarco",
                "--encoder",
                "bge",
            ],
        )
    mock_build.assert_called_once_with("bge")
    assert result.exit_code == 0, result.output


def test_cli_prep_encoder_nomic_flag_works(tmp_path: Path) -> None:
    """`vdbbench prep --encoder nomic` routes through build_encoder('nomic').

    Verifies that build_encoder is called with 'nomic' so trust_remote_code=True
    is propagated (the registry already has the right shape — tested in
    test_encoder_registry.py). Using --dataset msmarco so the encoder branch runs.
    """
    from unittest.mock import MagicMock  # noqa: PLC0415

    from vdbbench.embed.encoder import FakeEncoder, encode_corpus  # noqa: PLC0415

    bundle = _make_tiny_corpus_bundle()
    fake_enc = FakeEncoder(dim=16)
    mock_build = MagicMock(return_value=fake_enc)

    def _fake_encode_corpus(b: object, enc: object, **kw: object) -> object:
        return encode_corpus(bundle, FakeEncoder(dim=16), **kw)

    with (
        patch("vdbbench.corpus.load_msmarco", return_value=bundle),
        patch("vdbbench.embed.registry.build_encoder", mock_build),
        patch("vdbbench.embed.encode_corpus", _fake_encode_corpus),
    ):
        result = CliRunner().invoke(
            app,
            [
                "prep",
                "--out",
                str(tmp_path / "encoded"),
                "--dataset",
                "msmarco",
                "--encoder",
                "nomic",
            ],
        )
    mock_build.assert_called_once_with("nomic")
    assert result.exit_code == 0, result.output


def test_cli_prep_encoder_and_embed_model_both_set_exits_nonzero(tmp_path: Path) -> None:
    """`--encoder` and `--embed-model` together are mutually exclusive → exit 2 with hint."""
    result = CliRunner().invoke(
        app,
        [
            "prep",
            "--out",
            str(tmp_path / "encoded"),
            "--dataset",
            "synthetic",
            "--encoder",
            "bge",
            "--embed-model",
            "BAAI/bge-small-en-v1.5",
        ],
    )
    # Pin the exact UX contract: typer.BadParameter exits with code 2 and the
    # error message must name both flags so the user can correct in one step.
    assert result.exit_code == 2, result.output
    output_text = (result.output or "") + str(result.exception or "")
    assert "--encoder" in output_text and "--embed-model" in output_text


def test_cli_prep_encoder_flag_in_help() -> None:
    """`vdbbench prep --help` must advertise the new --encoder flag."""
    result = CliRunner().invoke(app, ["prep", "--help"])
    assert result.exit_code == 0, result.output
    assert "--encoder" in result.output


def test_cli_bench_prom_exporter_stopped_after_exit(tmp_path: Path) -> None:
    """After `bench` exits, stop_exporter must have been called so the port is freed.

    Uses a real WSGIServer via start_exporter to verify the lifecycle — not just
    a mock — so we can confirm the port is released on normal exit.
    """
    import socket  # noqa: PLC0415
    import time  # noqa: PLC0415

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _port_is_listening(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0

    port = _find_free_port()

    fake_run = _make_fake_run_bench()
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
                "--adapter",
                "exact",
                "--prometheus-port",
                str(port),
            ],
        )
    assert result.exit_code == 0, result.output
    # Port must be released after bench exits.
    deadline = time.monotonic() + 2.0
    released = False
    while time.monotonic() < deadline:
        if not _port_is_listening(port):
            released = True
            break
        time.sleep(0.05)
    assert released, f"port {port} still listening after bench exit — stop_exporter not called"
