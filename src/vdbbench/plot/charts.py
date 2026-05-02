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

import warnings
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


def _pareto_filter(sub: pd.DataFrame) -> pd.DataFrame:
    """Filter `sub` to its recall/latency Pareto frontier.

    A point is dominated when some other point has both **higher or
    equal** recall *and* **lower or equal** latency, with at least one
    of those inequalities strict. We keep only the non-dominated points.

    Tie-breaking: equal-recall, lower-latency wins. The previous
    implementation just sorted by recall and connected every point,
    leaving dominated points on the published frontier.
    """
    if sub.empty:
        return sub
    # Sort by recall descending, latency ascending — equal-recall ties
    # then prefer the lower-latency row, so a forward sweep can simply
    # keep the running minimum latency.
    df = sub.sort_values(
        by=["recall_at_k_mean", "latency_ms_p95"],
        ascending=[False, True],
    ).reset_index(drop=True)
    keep_idx: list[int] = []
    best_lat = float("inf")
    for i, row in df.iterrows():
        lat = float(row["latency_ms_p95"])
        # We're walking high-recall-first; a row survives only if it
        # offers strictly better latency than every higher-recall row
        # we've already kept. Otherwise some prior row dominates it.
        if lat < best_lat:
            keep_idx.append(int(i))
            best_lat = lat
    # Plot left-to-right by recall ascending so the curve still reads as
    # "more recall = more latency" without zig-zagging on ties.
    return df.iloc[keep_idx].sort_values("recall_at_k_mean").reset_index(drop=True)


def plot_pareto_frontier(summary: pd.DataFrame, out: str | Path) -> tuple[Path, Path]:
    """Recall@k (x) vs p95 latency (y), one Pareto curve per DB.

    For each DB we filter out dominated points first (a point is
    dominated when some other point has higher-or-equal recall AND
    lower-or-equal latency, strictly better in at least one) and then
    connect the survivors left-to-right by recall. The previous
    implementation sorted points by recall and connected every one of
    them, which left dominated points on the published frontier.
    """
    out_path = _ensure_outdir(out)
    fig, ax = plt.subplots(figsize=(8, 6))
    for db in sorted(summary["db"].unique()):
        sub = summary[summary["db"] == db]
        frontier = _pareto_filter(sub)
        ax.plot(
            frontier["recall_at_k_mean"],
            frontier["latency_ms_p95"],
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


def plot_memory_vs_recall(summary: pd.DataFrame, out: str | Path) -> tuple[Path, Path]:
    """Scatter plot: peak RSS (y) vs recall@k mean (x), one mark per (db, params).

    Honest companion to ``plot_pareto_frontier`` for the latency-vs-memory
    tradeoff. Falls back gracefully when memory columns are missing or
    all zero (e.g. older summaries from before memory sampling landed)
    by emitting an empty axes with an annotation, so ``plot_all`` doesn't
    have to gate the call.
    """
    out_path = _ensure_outdir(out)
    fig, ax = plt.subplots(figsize=(8, 6))
    if "peak_rss_bytes" not in summary.columns or float(summary["peak_rss_bytes"].max()) == 0.0:
        ax.text(
            0.5,
            0.5,
            "no peak_rss_bytes data in summary\n(re-run bench with the memory-aware harness)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
    else:
        for db in sorted(summary["db"].unique()):
            sub = summary[summary["db"] == db]
            ax.scatter(
                sub["recall_at_k_mean"],
                sub["peak_rss_bytes"] / (1024 * 1024),
                label=db,
                color=_DB_COLORS.get(db),
                s=60,
            )
        ax.set_xlabel("Recall@k (mean)")
        ax.set_ylabel("Peak RSS (MiB)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Memory vs recall — bench-process peak RSS per (db, params)")
        ax.legend()
    fig.tight_layout()
    paths = _save_both(fig, out_path, "memory_recall")
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
    # `set_yscale("log")` warns and silently falls back when all values are
    # ≤ 0 (e.g. ingest=0 for embedded adapters with instant ingest, or
    # bytes_disk=0 for in-memory references). Keep linear in that case.
    if log and any(v is not None and v > 0 for v in values):
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    paths = _save_both(fig, out_path, name)
    plt.close(fig)
    return paths


def plot_speedup_vs_baseline(
    summary: pd.DataFrame,
    out: str | Path,
    *,
    baseline_db: str | None = "chroma",
) -> tuple[Path, Path]:
    """Per-DB speedup over `baseline_db` p95 latency.

    Speedup = baseline_p95 / db_p95. >1 means faster than baseline. The
    baseline itself is shown as a 1.0 reference bar so the chart still
    makes sense when chroma isn't in the run.

    Baseline resolution:

    * If `baseline_db` is given and present, use it.
    * If it is given but missing from the summary, raise — silently
      falling back to "alphabetically first" used to mask reviewer typos
      and produce a chart titled with a DB that wasn't actually used.
    * If `baseline_db` is `None`, warn and pick the alphabetically-first
      DB so the call still produces a chart for ad-hoc plotting.
    """
    out_path = _ensure_outdir(out)
    available = sorted(set(summary["db"]))
    if baseline_db is not None and baseline_db not in summary["db"].values:
        raise ValueError(
            f"baseline_db={baseline_db!r} not found in summary; available: {available}"
        )
    if baseline_db is None:
        baseline_db = available[0]
        warnings.warn(
            f"plot_speedup_vs_baseline: baseline_db not given; "
            f"defaulting to alphabetically-first DB {baseline_db!r}",
            stacklevel=2,
        )
    baseline_lat = float(summary[summary["db"] == baseline_db]["latency_ms_p95"].mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    speedups = (baseline_lat / summary["latency_ms_p95"]).tolist()
    labels = summary["label"].tolist()
    colors = [_DB_COLORS.get(db, "#666666") for db in summary["db"]]
    ax.bar(range(len(labels)), speedups, color=colors)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Speedup vs {baseline_db} (p95 latency, higher = faster)")
    ax.set_title(f"p95 latency speedup relative to {baseline_db}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    paths = _save_both(fig, out_path, "speedup")
    plt.close(fig)
    return paths


def plot_all(summary_path: str | Path, out: str | Path) -> dict[str, tuple[Path, Path]]:
    """Read `summary.parquet` and emit every standard chart into `out/`.

    `chroma` is the preferred speedup baseline (it's the simplest in-memory
    adapter, so its p95 is a reasonable "no tuning" floor). If it isn't in
    the run, fall back to `None` so `plot_speedup_vs_baseline` warns and
    picks the alphabetically-first DB rather than raising — `plot_all` is
    the "produce everything you can" entry point and should degrade
    gracefully.
    """
    summary = pd.read_parquet(summary_path)
    if summary.empty:
        raise ValueError(f"summary at {summary_path} is empty")
    speedup_baseline: str | None = "chroma" if "chroma" in summary["db"].values else None
    return {
        "pareto": plot_pareto_frontier(summary, out),
        "recall": plot_axis_bars(
            summary,
            out,
            column="recall_at_k_mean",
            title="Recall@k (mean) — higher is better",
            ylabel="Recall@k",
            name="recall",
        ),
        "latency": plot_axis_bars(
            summary,
            out,
            column="latency_ms_p95",
            title="p95 query latency (lower is better)",
            ylabel="latency (ms, log scale)",
            name="latency",
            log=True,
        ),
        "ingest": plot_axis_bars(
            summary,
            out,
            column="ingest_throughput_vps",
            title="Ingest throughput (higher is better)",
            ylabel="vectors / sec (log scale)",
            name="ingest",
            log=True,
        ),
        "index_disk": plot_axis_bars(
            summary,
            out,
            column="index_bytes",
            title="Index disk footprint (lower is better)",
            ylabel="bytes (log scale)",
            name="index_disk",
            log=True,
        ),
        "speedup": plot_speedup_vs_baseline(summary, out, baseline_db=speedup_baseline),
        "memory_recall": plot_memory_vs_recall(summary, out),
    }
