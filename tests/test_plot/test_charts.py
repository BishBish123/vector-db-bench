"""Smoke tests for chart generation — verify shape, not aesthetics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from vdbbench.plot.charts import (
    _pareto_filter,
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
        # a legit case in real data too (chroma not in the run); plot_all
        # picks the alphabetically-first DB itself instead of letting
        # plot_speedup_vs_baseline emit a UserWarning. Assert no warning
        # fires through this entry point so the CLI / smoke runs stay
        # quiet for non-chroma summaries.
        import warnings as _warnings  # noqa: PLC0415

        df = _toy_summary()
        path = tmp_path / "summary.parquet"
        df.to_parquet(path, index=False)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            result = plot_all(path, tmp_path / "out")
        for name in (
            "pareto",
            "recall",
            "latency",
            "ingest",
            "index_disk",
            "speedup",
            "memory_recall",
        ):
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

    def test_pareto_filters_dominated_points(self) -> None:
        """A point with both lower recall AND higher latency is dominated
        and must not appear on the Pareto frontier curve.
        """
        df = pd.DataFrame(
            [
                # Pareto-optimal: high recall, low latency.
                {"db": "x", "recall_at_k_mean": 0.9, "latency_ms_p95": 5.0},
                # Pareto-optimal: lower recall but even lower latency.
                {"db": "x", "recall_at_k_mean": 0.7, "latency_ms_p95": 1.0},
                # Dominated by both above (worse recall AND worse latency
                # than the 0.9/5.0 point).
                {"db": "x", "recall_at_k_mean": 0.7, "latency_ms_p95": 20.0},
                # Dominated by the 0.7/1.0 point (same recall, higher latency).
                {"db": "x", "recall_at_k_mean": 0.7, "latency_ms_p95": 4.0},
            ]
        )
        frontier = _pareto_filter(df)
        kept = list(zip(frontier["recall_at_k_mean"], frontier["latency_ms_p95"], strict=True))
        # Two non-dominated points; sorted left-to-right by recall ascending.
        assert kept == [(0.7, 1.0), (0.9, 5.0)]

    def test_pareto_tie_break_prefers_lower_latency(self) -> None:
        """Equal-recall ties go to the lower-latency point."""
        df = pd.DataFrame(
            [
                {"db": "x", "recall_at_k_mean": 0.8, "latency_ms_p95": 10.0},
                {"db": "x", "recall_at_k_mean": 0.8, "latency_ms_p95": 3.0},  # winner
                {"db": "x", "recall_at_k_mean": 0.8, "latency_ms_p95": 7.0},
            ]
        )
        frontier = _pareto_filter(df)
        assert len(frontier) == 1
        assert float(frontier.iloc[0]["latency_ms_p95"]) == 3.0


def _multi_config_summary() -> pd.DataFrame:
    """Two chroma configs + one qdrant config — a realistic multi-config run."""
    rows: list[dict[str, object]] = []
    for label, p95 in (
        ("chroma:default", 4.0),
        ("chroma:hnsw-tuned", 8.0),
    ):
        rows.append(
            {
                "db": "chroma",
                "label": label,
                "params_hash": label.split(":", 1)[1],
                "params_json": "{}",
                "n_passages": 1000,
                "n_queries": 100,
                "dim": 16,
                "ingest_s": 1.0,
                "ingest_throughput_vps": 1000.0,
                "index_s": 0.5,
                "index_bytes": 1024,
                "latency_ms_mean": p95 * 0.5,
                "latency_ms_p50": p95 * 0.4,
                "latency_ms_p95": p95,
                "recall_at_k_mean": 0.9,
                "recall_at_k_p50": 0.9,
                "ndcg_at_k_mean": 0.85,
                "qps_estimate": 1000.0 / p95,
            }
        )
    rows.append(
        {
            "db": "qdrant",
            "label": "qdrant:hnsw-default",
            "params_hash": "qhash",
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
            "latency_ms_p95": 2.0,
            "recall_at_k_mean": 0.95,
            "recall_at_k_p50": 0.95,
            "ndcg_at_k_mean": 0.92,
            "qps_estimate": 500.0,
        }
    )
    return pd.DataFrame(rows)


class TestSpeedupBaseline:
    def test_speedup_resolves_single_config_baseline(self, tmp_path: Path) -> None:
        """One chroma config + one qdrant config — chroma row auto-selected,
        no `baseline_label` needed."""
        df = _toy_summary().copy()
        # Mutate pgvector row to chroma so we have a 1-config baseline DB.
        df.loc[df["db"] == "pgvector", "db"] = "chroma"
        df.loc[df["db"] == "chroma", "label"] = "chroma:default"
        png, svg = plot_speedup_vs_baseline(df, tmp_path, baseline_db="chroma")
        assert png.exists() and svg.exists()

    def test_speedup_requires_label_with_multi_config_baseline(self, tmp_path: Path) -> None:
        """Two chroma configs, no baseline_label — must raise with both
        labels listed so the caller can fix the call."""
        df = _multi_config_summary()
        with pytest.raises(ValueError, match="2 configs"):
            plot_speedup_vs_baseline(df, tmp_path, baseline_db="chroma")
        # And the error message must list the available labels.
        with pytest.raises(ValueError, match="chroma:default") as exc:
            plot_speedup_vs_baseline(df, tmp_path, baseline_db="chroma")
        assert "chroma:hnsw-tuned" in str(exc.value)

    def test_speedup_baseline_renders_as_one(self, tmp_path: Path) -> None:
        """The chosen baseline row's bar must be exactly 1.0, not the
        result of float division of equal numbers (which is fine in
        practice but loses meaning when the chart is the visual claim).
        """
        df = _multi_config_summary()
        # Inspect the speedups via a stubbed `bar` to capture the values.
        captured: dict[str, list[float]] = {}

        def fake_bar(_self, _x, heights, **_kwargs):  # type: ignore[no-untyped-def]
            captured["heights"] = list(heights)

        with patch("matplotlib.axes.Axes.bar", fake_bar):
            plot_speedup_vs_baseline(
                df,
                tmp_path,
                baseline_db="chroma",
                baseline_label="chroma:default",
            )
        # chroma:default row anchored at 1.0 exactly.
        labels = df["label"].tolist()
        baseline_idx = labels.index("chroma:default")
        assert captured["heights"][baseline_idx] == 1.0
        # qdrant:hnsw-default — p95 2.0 vs baseline 4.0 -> speedup 2.0.
        qdrant_idx = labels.index("qdrant:hnsw-default")
        assert captured["heights"][qdrant_idx] == pytest.approx(2.0)

    def test_speedup_unknown_label_lists_options(self, tmp_path: Path) -> None:
        """Typo in baseline_label surfaces a list of available labels."""
        df = _multi_config_summary()
        with pytest.raises(ValueError, match="not found among rows") as exc:
            plot_speedup_vs_baseline(
                df,
                tmp_path,
                baseline_db="chroma",
                baseline_label="chroma:typo",
            )
        assert "chroma:default" in str(exc.value)
        assert "chroma:hnsw-tuned" in str(exc.value)
