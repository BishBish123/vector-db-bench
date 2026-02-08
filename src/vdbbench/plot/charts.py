"""Chart generation from `summary.parquet`.

Charts produced:

* `pareto.{png,svg}` — recall@k vs p95 latency, one curve per DB. Centerpiece
  of the blog post.
* `recall.{png,svg}` — bars: recall@k mean per DB / param-hash.
* `latency.{png,svg}` — bars: p95 latency per DB / param-hash.
* `ingest.{png,svg}` — bars: ingest throughput vectors/sec per DB.
* `index_disk.{png,svg}` — bars: index size on disk per DB.

All charts read from a single `summary.parquet` produced by `vdbbench.bench`,
so re-running `make plots` after a bench is cheap and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

# Stable per-DB colors so charts read consistently across runs.
_DB_COLORS: dict[str, str] = {
    "pgvector": "#336791",  # Postgres blue
    "qdrant": "#dc382d",  # Qdrant red
    "lancedb": "#1f77b4",
    "chroma": "#ff7f0e",
    "mem": "#999999",
}


def _ensure_outdir(out: str | Path) -> Path:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_both(fig: matplotlib.figure.Figure, out: Path, name: str) -> tuple[Path, Path]:
    png = out / f"{name}.png"
    svg = out / f"{name}.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_pareto_frontier(summary: pd.DataFrame, out: str | Path) -> tuple[Path, Path]:
    """Recall@k (x) vs p95 latency (y), one connected curve per DB.

    Within each DB, points are sorted by recall ascending so the curve
    actually traces a monotonic frontier when the bench sweep covers the
    accuracy/latency tradeoff knobs (e.g. HNSW ef_search).
    """
    out_path = _ensure_outdir(out)
    fig, ax = plt.subplots(figsize=(8, 6))
    for db in sorted(summary["db"].unique()):
        sub = summary[summary["db"] == db].sort_values("recall_at_k_mean")
        ax.plot(
            sub["recall_at_k_mean"],
            sub["latency_ms_p95"],
            marker="o",
            linewidth=1.5,
            label=db,
            color=_DB_COLORS.get(db),
        )
    ax.set_xlabel("Recall@k (mean)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_title("Vector DB Pareto frontier — recall vs p95 latency")
    ax.legend()
    fig.tight_layout()
    paths = _save_both(fig, out_path, "pareto")
    plt.close(fig)
    return paths


def plot_axis_bars(
    summary: pd.DataFrame,
    out: str | Path,
    *,
    column: str,
    title: str,
    ylabel: str,
    name: str,
    log: bool = False,
) -> tuple[Path, Path]:
    """One bar per (db, params_hash). Data values are sorted within each db."""
    out_path = _ensure_outdir(out)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = summary["label"].tolist()
    values = summary[column].tolist()
    colors = [_DB_COLORS.get(db, "#666666") for db in summary["db"]]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    paths = _save_both(fig, out_path, name)
    plt.close(fig)
    return paths


def plot_all(summary_path: str | Path, out: str | Path) -> dict[str, tuple[Path, Path]]:
    """Read `summary.parquet` and emit every standard chart into `out/`."""
    summary = pd.read_parquet(summary_path)
    if summary.empty:
        raise ValueError(f"summary at {summary_path} is empty")
    return {
        "pareto": plot_pareto_frontier(summary, out),
        "recall": plot_axis_bars(
            summary,
            out,
            column="recall_at_k_mean",
            title="Recall@k (mean)",
            ylabel="Recall",
            name="recall",
        ),
        "latency": plot_axis_bars(
            summary,
            out,
            column="latency_ms_p95",
            title="p95 query latency",
            ylabel="ms",
            name="latency",
            log=True,
        ),
        "ingest": plot_axis_bars(
            summary,
            out,
            column="ingest_throughput_vps",
            title="Ingest throughput",
            ylabel="vectors / sec",
            name="ingest",
            log=True,
        ),
        "index_disk": plot_axis_bars(
            summary,
            out,
            column="index_bytes",
            title="Index disk footprint",
            ylabel="bytes",
            name="index_disk",
            log=True,
        ),
    }
