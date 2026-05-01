"""Smoke tests for chart generation — verify shape, not aesthetics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vdbbench.plot.charts import (
    plot_all,
    plot_axis_bars,
    plot_pareto_frontier,
    plot_speedup_vs_baseline,
)


def _toy_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "db": "pgvector",
                "label": "pgvector:hash1",
                "params_hash": "hash1",
                "params_json": "{}",
                "n_passages": 1000,
                "n_queries": 100,
                "dim": 16,
                "ingest_s": 1.0,
                "ingest_throughput_vps": 1000.0,
                "index_s": 0.5,
                "index_bytes": 1024,
                "latency_ms_mean": 2.0,
                "latency_ms_p50": 1.5,
                "latency_ms_p95": 5.0,
                "recall_at_k_mean": 0.9,
                "recall_at_k_p50": 0.9,
                "ndcg_at_k_mean": 0.85,
                "qps_estimate": 500.0,
            },
            {
                "db": "qdrant",
                "label": "qdrant:hash2",
                "params_hash": "hash2",
                "params_json": "{}",
                "n_passages": 1000,
                "n_queries": 100,
                "dim": 16,
                "ingest_s": 0.5,
                "ingest_throughput_vps": 2000.0,
                "index_s": 0.0,
                "index_bytes": 2048,
                "latency_ms_mean": 1.0,
                "latency_ms_p50": 0.8,
                "latency_ms_p95": 2.5,
                "recall_at_k_mean": 0.95,
                "recall_at_k_p50": 0.95,
                "ndcg_at_k_mean": 0.92,
                "qps_estimate": 1000.0,
            },
        ]
    )


class TestCharts:
    def test_pareto_emits_png_and_svg(self, tmp_path: Path) -> None:
        png, svg = plot_pareto_frontier(_toy_summary(), tmp_path)
        assert png.exists() and png.stat().st_size > 0
        assert svg.exists() and svg.stat().st_size > 0

    def test_axis_bars_emits_png_and_svg(self, tmp_path: Path) -> None:
        png, svg = plot_axis_bars(
            _toy_summary(),
            tmp_path,
            column="recall_at_k_mean",
            title="recall",
            ylabel="recall",
            name="recall",
        )
        assert png.exists() and svg.exists()

    def test_plot_all_writes_every_chart(self, tmp_path: Path) -> None:
        # Toy summary doesn't include the default `chroma` baseline. That's
        # a legit case in real data too (chroma not in the run), and
        # `plot_all` is documented to degrade by warning + picking
        # alphabetically. Pin that path so the test surfaces a regression
        # if the fallback ever flips back to silent.
        df = _toy_summary()
        path = tmp_path / "summary.parquet"
        df.to_parquet(path, index=False)
        with pytest.warns(UserWarning, match="alphabetically-first"):
            result = plot_all(path, tmp_path / "out")
        for name in ("pareto", "recall", "latency", "ingest", "index_disk", "speedup"):
            png, svg = result[name]
            assert png.exists() and svg.exists()

    def test_plot_all_rejects_empty_summary(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.parquet"
        pd.DataFrame(columns=_toy_summary().columns).to_parquet(empty, index=False)
        with pytest.raises(ValueError, match="empty"):
            plot_all(empty, tmp_path / "out")

    def test_speedup_chart_with_explicit_baseline(self, tmp_path: Path) -> None:
        png, svg = plot_speedup_vs_baseline(_toy_summary(), tmp_path, baseline_db="pgvector")
        assert png.exists() and svg.exists()

    def test_speedup_chart_raises_when_named_baseline_missing(self, tmp_path: Path) -> None:
        """If a baseline is explicitly named but not present in the data,
        raise — silently substituting the alphabetically-first DB used to
        let typos through and produce a chart titled against a DB that
        wasn't even in the run.
        """
        df = _toy_summary()  # only pgvector + qdrant
        with pytest.raises(ValueError, match="not found in summary"):
            plot_speedup_vs_baseline(df, tmp_path, baseline_db="chroma")

    def test_speedup_chart_warns_when_baseline_is_none(self, tmp_path: Path) -> None:
        """`baseline_db=None` is the explicit "pick something for me" path,
        so we warn (don't raise) and still emit a chart."""
        df = _toy_summary()
        with pytest.warns(UserWarning, match="alphabetically-first"):
            png, svg = plot_speedup_vs_baseline(df, tmp_path, baseline_db=None)
        assert png.exists() and svg.exists()
