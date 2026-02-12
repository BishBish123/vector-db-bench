"""Plot generation from a bench summary parquet."""

from vdbbench.plot.charts import (
    plot_all,
    plot_axis_bars,
    plot_pareto_frontier,
    plot_speedup_vs_baseline,
)

__all__ = [
    "plot_all",
    "plot_axis_bars",
    "plot_pareto_frontier",
    "plot_speedup_vs_baseline",
]
