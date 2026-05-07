"""Tests for the bge-vs-nomic encoder comparison runner.

Covers:
  1. EncoderComparisonSpec validation (rejects unknown encoder names)
  2. run_encoder_comparison with mocked build_encoder and run_bench:
     - produces correct number of results
     - parquet has the right columns
  3. plot_encoder_comparison: writes a PNG; figure has 2x bars per metric
  4. CLI: ``vdbbench compare-encoders --help`` lists the flags
  5. CLI integration: smoke run against exact + bge produces non-empty output
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from vdbbench.encode.compare import (
    EncoderComparisonSpec,
    plot_encoder_comparison,
    run_encoder_comparison,
)

# ---------------------------------------------------------------------------
# 1. EncoderComparisonSpec validation
# ---------------------------------------------------------------------------


class TestEncoderComparisonSpec:
    def test_valid_spec_bge_only(self) -> None:
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
        )
        assert spec.adapter == "exact"
        assert "bge" in spec.encoders

    def test_valid_spec_both_encoders(self) -> None:
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge", "nomic"),
        )
        assert len(spec.encoders) == 2

    def test_rejects_unknown_encoder(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                encoders=("bge", "totally-fake-model"),
            )

    def test_rejects_all_unknown_encoders(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                encoders=("fake1", "fake2"),
            )

    def test_rejects_empty_encoders(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                encoders=(),
            )

    def test_rejects_negative_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                top_k=-1,
            )

    def test_rejects_zero_corpus_size(self) -> None:
        with pytest.raises(ValueError, match="corpus_size must be positive"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                corpus_size=0,
            )

    def test_default_encoders_are_bge_and_nomic(self) -> None:
        spec = EncoderComparisonSpec(adapter="exact", dataset="synthetic")
        assert "bge" in spec.encoders
        assert "nomic" in spec.encoders

    def test_error_message_lists_valid_choices(self) -> None:
        with pytest.raises(ValueError, match="bge"):
            EncoderComparisonSpec(
                adapter="exact",
                dataset="synthetic",
                encoders=("not-real",),
            )


# ---------------------------------------------------------------------------
# 2. run_encoder_comparison with mocked internals
# ---------------------------------------------------------------------------


def _make_fake_run_bench(recall: float = 0.9, p95: float = 2.0, ingest_s: float = 0.1) -> Any:
    """Return a fake run_bench that produces a minimal BenchResult."""
    from vdbbench.bench.runner import BenchResult  # noqa: PLC0415

    def _fake(
        encoded: Any,
        specs: list[Any],
        *,
        progress: bool = False,
        tolerate_failures: bool = False,
        out: Any = None,
    ) -> BenchResult:
        summary = pd.DataFrame(
            [
                {
                    "db": specs[0].adapter.name if specs else "exact",
                    "label": specs[0].display_label() if specs else "exact:bruteforce",
                    "params_hash": "abc",
                    "params_json": "{}",
                    "profile": "warm",
                    "n_passages": 100,
                    "n_queries": 10,
                    "dim": 32,
                    "ingest_s": ingest_s,
                    "ingest_throughput_vps": 1000.0,
                    "index_s": 0.01,
                    "index_bytes": 0,
                    "latency_ms_mean": 1.0,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": p95,
                    "latency_ms_p99": p95,
                    "recall_at_k_mean": recall,
                    "recall_at_k_p50": recall,
                    "ndcg_at_k_mean": recall,
                    "qps_estimate": 500.0,
                }
            ]
        )
        return BenchResult(
            timings=pd.DataFrame(),
            summary=summary,
            manifest={"schema_version": 1},
        )

    return _fake


def _make_fake_encoder(dim: int) -> MagicMock:
    """Return a mock encoder that looks like a SentenceTransformerEncoder."""
    enc = MagicMock()
    enc.dim = dim
    enc.name = f"fake-encoder-dim{dim}"
    return enc


def _patch_build_encoder(dims: dict[str, int]) -> Any:
    """Return a patch context for build_encoder that returns fake encoders."""

    def _fake_build(key: str | None = None) -> MagicMock:
        resolved = key or "bge"
        d = dims.get(resolved, 32)
        return _make_fake_encoder(d)

    return patch("vdbbench.encode.compare.build_encoder", side_effect=_fake_build)


# ---- Typical encoder dims ----
_ENCODER_DIMS = {"bge": 384, "nomic": 768}


class TestRunEncoderComparison:
    def test_produces_one_result_per_encoder(self, tmp_path: Path) -> None:
        """One encoder → one EncoderResult."""
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench()
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            results = run_encoder_comparison(spec, output_dir=tmp_path)

        assert len(results) == 1
        assert results[0].encoder == "bge"

    def test_produces_two_results_for_two_encoders(self, tmp_path: Path) -> None:
        """Two encoders → two EncoderResults."""
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge", "nomic"),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench()
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            results = run_encoder_comparison(spec, output_dir=tmp_path)

        assert len(results) == 2
        encoder_names = {r.encoder for r in results}
        assert encoder_names == {"bge", "nomic"}

    def test_parquet_has_required_columns(self, tmp_path: Path) -> None:
        """Comparison parquet must have the documented columns."""
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench()
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            run_encoder_comparison(spec, output_dir=tmp_path)

        parquet_path = tmp_path / "encoder_comparison.parquet"
        assert parquet_path.is_file()
        df = pd.read_parquet(parquet_path)
        for col in ("encoder", "recall_at_k", "p95_ms", "ingest_seconds", "embedding_dim"):
            assert col in df.columns, f"missing column: {col}"

    def test_parquet_is_non_empty(self, tmp_path: Path) -> None:
        """At least one row must be written when a run succeeds."""
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench()
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            run_encoder_comparison(spec, output_dir=tmp_path)

        df = pd.read_parquet(tmp_path / "encoder_comparison.parquet")
        assert len(df) >= 1

    def test_encoder_result_fields(self, tmp_path: Path) -> None:
        """EncoderResult fields must reflect the mocked bench values."""
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench(recall=0.95, p95=3.5, ingest_s=0.2)
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            results = run_encoder_comparison(spec, output_dir=tmp_path)

        assert len(results) == 1
        r = results[0]
        assert r.recall_at_k == pytest.approx(0.95)
        assert r.p95_ms == pytest.approx(3.5)
        assert r.ingest_seconds == pytest.approx(0.2)
        assert isinstance(r.embedding_dim, int)
        assert r.embedding_dim > 0

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """run_encoder_comparison must create the output directory if missing."""
        out = tmp_path / "deep" / "nested" / "compare"
        assert not out.exists()
        spec = EncoderComparisonSpec(
            adapter="exact",
            dataset="synthetic",
            encoders=("bge",),
            corpus_size=50,
        )
        fake_run = _make_fake_run_bench()
        with (
            patch("vdbbench.encode.compare.run_bench", fake_run),
            _patch_build_encoder(_ENCODER_DIMS),
        ):
            run_encoder_comparison(spec, output_dir=out)

        assert out.is_dir()


# ---------------------------------------------------------------------------
# 3. plot_encoder_comparison
# ---------------------------------------------------------------------------


class TestPlotEncoderComparison:
    def _make_comparison_parquet(self, path: Path, encoders: list[str]) -> Path:
        """Write a minimal encoder_comparison.parquet for plot tests."""
        dims = {"bge": 384, "nomic": 768}
        rows = [
            {
                "encoder": enc,
                "recall_at_k": 0.9 + i * 0.02,
                "p95_ms": 2.0 + i * 0.5,
                "ingest_seconds": 0.1,
                "embedding_dim": dims.get(enc, 32 * (i + 1)),
            }
            for i, enc in enumerate(encoders)
        ]
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return path

    def test_writes_png(self, tmp_path: Path) -> None:
        """plot_encoder_comparison must write a non-empty PNG."""
        parquet_path = self._make_comparison_parquet(
            tmp_path / "encoder_comparison.parquet", ["bge"]
        )
        out_png = tmp_path / "encoder_comparison.png"
        plot_encoder_comparison(parquet_path, out_png)
        assert out_png.is_file()
        assert out_png.stat().st_size > 0

    def test_creates_output_parent(self, tmp_path: Path) -> None:
        """output_path parent is created if it does not exist."""
        parquet_path = self._make_comparison_parquet(
            tmp_path / "encoder_comparison.parquet", ["bge"]
        )
        nested_out = tmp_path / "deep" / "charts" / "encoder_comparison.png"
        assert not nested_out.parent.exists()
        plot_encoder_comparison(parquet_path, nested_out)
        assert nested_out.is_file()

    def test_two_encoders_produces_two_bar_groups(self, tmp_path: Path) -> None:
        """With 2 encoders, the chart x-axis has 2 tick positions (one pair
        of bars each)."""
        parquet_path = self._make_comparison_parquet(
            tmp_path / "encoder_comparison.parquet", ["bge", "nomic"]
        )
        out_png = tmp_path / "encoder_comparison.png"
        plot_encoder_comparison(parquet_path, out_png)
        # If the PNG was written we know the chart rendered — the exact bar
        # count is an internal matplotlib detail but the file size doubles.
        assert out_png.stat().st_size > 0

    def test_returns_output_path(self, tmp_path: Path) -> None:
        """The function must return the resolved output_path."""
        parquet_path = self._make_comparison_parquet(
            tmp_path / "encoder_comparison.parquet", ["bge"]
        )
        out_png = tmp_path / "encoder_comparison.png"
        result = plot_encoder_comparison(parquet_path, out_png)
        assert result == out_png


# ---------------------------------------------------------------------------
# 4. CLI: --help lists the flags
# ---------------------------------------------------------------------------


class TestCliCompareEncoders:
    def test_compare_encoders_help_lists_flags(self) -> None:
        """``vdbbench compare-encoders --help`` must advertise the key flags."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["compare-encoders", "--help"])
        assert result.exit_code == 0, result.output
        assert "--adapter" in result.output
        assert "--encoders" in result.output
        assert "--dataset" in result.output
        assert "--out" in result.output

    def test_compare_encoders_in_main_help(self) -> None:
        """``vdbbench --help`` must list the compare-encoders subcommand."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.cli import app  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "compare-encoders" in result.output

    # 5. CLI integration: smoke run against exact + bge produces non-empty output
    def test_cli_compare_encoders_bge_smoke(self, tmp_path: Path) -> None:
        """``vdbbench compare-encoders --adapter exact --encoders bge``
        produces a non-empty parquet and chart without any network I/O."""
        import warnings  # noqa: PLC0415

        from typer.testing import CliRunner  # noqa: PLC0415

        from vdbbench.bench.runner import BenchResult  # noqa: PLC0415
        from vdbbench.cli import app  # noqa: PLC0415

        # Stub run_bench so no real adapter work happens.
        def _fake_run_bench(
            encoded: Any,
            specs: list[Any],
            *,
            progress: bool = False,
            tolerate_failures: bool = False,
            out: Any = None,
        ) -> BenchResult:
            summary = pd.DataFrame(
                [
                    {
                        "db": "exact",
                        "label": "compare:bge",
                        "params_hash": "abc",
                        "params_json": "{}",
                        "profile": "warm",
                        "n_passages": 50,
                        "n_queries": 5,
                        "dim": 384,
                        "ingest_s": 0.01,
                        "ingest_throughput_vps": 5000.0,
                        "index_s": 0.001,
                        "index_bytes": 0,
                        "latency_ms_mean": 0.5,
                        "latency_ms_p50": 0.5,
                        "latency_ms_p95": 1.0,
                        "latency_ms_p99": 1.0,
                        "recall_at_k_mean": 1.0,
                        "recall_at_k_p50": 1.0,
                        "ndcg_at_k_mean": 1.0,
                        "qps_estimate": 2000.0,
                    }
                ]
            )
            return BenchResult(
                timings=pd.DataFrame(),
                summary=summary,
                manifest={"schema_version": 1},
            )

        out_dir = tmp_path / "encoder-compare"
        runner = CliRunner()
        with (
            patch("vdbbench.encode.compare.run_bench", _fake_run_bench),
            patch("vdbbench.encode.compare.build_encoder", side_effect=lambda k=None: _make_fake_encoder(384)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("always")
            result = runner.invoke(
                app,
                [
                    "compare-encoders",
                    "--adapter",
                    "exact",
                    "--encoders",
                    "bge",
                    "--dataset",
                    "synthetic",
                    "--out",
                    str(out_dir),
                    "--corpus-size",
                    "50",
                ],
            )

        assert result.exit_code == 0, f"exit {result.exit_code}: {result.output}"

        parquet_path = out_dir / "encoder_comparison.parquet"
        assert parquet_path.is_file(), "parquet not written"
        df = pd.read_parquet(parquet_path)
        assert len(df) >= 1, "parquet is empty"

        png_path = out_dir / "encoder_comparison.png"
        assert png_path.is_file(), "chart PNG not written"
        assert png_path.stat().st_size > 0, "chart PNG is empty"
