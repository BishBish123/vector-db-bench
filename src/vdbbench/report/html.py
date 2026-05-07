"""HTML report renderer — turns bench parquet + manifest + charts into a single-page HTML.

Usage::

    from vdbbench.report.html import render_html_report
    render_html_report(
        parquet_path=Path("results/demo/summary.parquet"),
        manifest_path=Path("results/demo/bench_manifest.json"),
        output_path=Path("results/demo/report.html"),
        charts_dir=Path("assets"),
    )

Dependencies: stdlib only (html, json, base64, datetime, pathlib, subprocess).
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["render_html_report"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --accent: #4f6ef7;
    --bg: #f7f8fc;
    --card: #ffffff;
    --border: #e2e6ea;
    --text: #1a1e2e;
    --muted: #6b7280;
    --good: #16a34a;
    --warn: #ca8a04;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0 1rem 4rem;
}
.container { max-width: 960px; margin: 0 auto; }
header {
    background: var(--accent);
    color: #fff;
    padding: 2rem 1rem 1.5rem;
    margin: 0 -1rem 2rem;
}
header h1 { font-size: 1.8rem; font-weight: 700; }
header .meta { font-size: 0.85rem; opacity: 0.85; margin-top: 0.4rem; }
h2 {
    font-size: 1.2rem; font-weight: 600;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.3rem; margin: 2rem 0 1rem;
}
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;
}
dl.info-grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.35rem 1.5rem;
}
dl.info-grid dt { font-weight: 600; white-space: nowrap; }
dl.info-grid dd { color: var(--muted); word-break: break-word; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th {
    background: var(--accent); color: #fff;
    padding: 0.55rem 0.75rem; text-align: left;
    white-space: nowrap;
}
td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f1f3fb; }
.tag {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    background: #e8ecfd; color: var(--accent);
    border-radius: 4px; padding: 0.1rem 0.45rem;
}
.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 1.25rem;
    margin-top: 0.5rem;
}
.chart-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; text-align: center;
}
.chart-card img { max-width: 100%; height: auto; display: block; }
.chart-caption {
    font-size: 0.8rem; color: var(--muted);
    padding: 0.5rem; border-top: 1px solid var(--border);
}
footer {
    text-align: center; font-size: 0.8rem; color: var(--muted);
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
}
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }
"""


def _esc(val: object) -> str:
    """HTML-escape an arbitrary value."""
    return html.escape(str(val))


def _fmt_float(val: object, digits: int = 3) -> str:
    """Format a float to `digits` decimal places, or return 'N/A' if not a number."""
    try:
        return f"{float(val):.{digits}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"


def _fmt_bytes(b: int | float) -> str:
    """Format bytes to a human-readable string."""
    try:
        n = float(b)
    except (TypeError, ValueError):
        return "N/A"
    if n < 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _git_hash() -> str | None:
    """Return the short git HEAD hash, or None if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _embed_png_as_data_uri(path: Path) -> str:
    """Read a PNG file and return a data: URI string."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_header(manifest: dict[str, Any], generated_at: str, git_hash: str | None) -> str:
    vdbbench_version = _esc(manifest.get("vdbbench_version", "unknown"))
    started = _esc(manifest.get("bench_started_at", "unknown"))
    meta_parts = [f"Generated {_esc(generated_at)}", f"Bench started {started}"]
    if git_hash:
        meta_parts.append(f"git {_esc(git_hash)}")
    meta_str = " &nbsp;|&nbsp; ".join(meta_parts)
    return (
        "<header>\n"
        "  <div class='container'>\n"
        f"    <h1>vdbbench Report &mdash; v{vdbbench_version}</h1>\n"
        f"    <div class='meta'>{meta_str}</div>\n"
        "  </div>\n"
        "</header>\n"
    )


def _build_methodology(manifest: dict[str, Any]) -> str:
    host = manifest.get("host_metadata", {})
    adapter_versions = manifest.get("adapter_versions", {})
    encoder_name = _esc(manifest.get("encoder_name", "unknown"))
    encoder_dim = _esc(manifest.get("encoder_dim", "unknown"))
    total_ram = int(host.get("total_memory_bytes", 0))
    cpu_count = host.get("cpu_count", "unknown")
    platform = _esc(host.get("platform", "unknown"))
    python_version = _esc(host.get("python_version", "unknown"))

    # Build adapter version list
    adapter_rows = "".join(
        f"<dd><span class='tag'>{_esc(k)}</span> {_esc(v)}</dd>"
        for k, v in adapter_versions.items()
    )
    if not adapter_rows:
        adapter_rows = "<dd><em>none recorded</em></dd>"

    specs_html = ""
    bench_specs = manifest.get("bench_specs", [])
    if bench_specs:
        spec_tags = " ".join(f"<span class='tag'>{_esc(s.get('label',''))}</span>" for s in bench_specs)
        specs_html = f"<dt>Bench specs</dt><dd>{spec_tags}</dd>"

    return (
        "<h2>Methodology</h2>\n"
        "<div class='card'>\n"
        "  <dl class='info-grid'>\n"
        f"    <dt>Encoder</dt><dd>{encoder_name} (dim {encoder_dim})</dd>\n"
        f"    <dt>Adapter versions</dt>{adapter_rows}\n"
        f"    {specs_html}\n"
        f"    <dt>CPU cores</dt><dd>{_esc(cpu_count)}</dd>\n"
        f"    <dt>RAM</dt><dd>{_esc(_fmt_bytes(total_ram))}</dd>\n"
        f"    <dt>Platform</dt><dd>{platform}</dd>\n"
        f"    <dt>Python</dt><dd>{python_version}</dd>\n"
        "  </dl>\n"
        "</div>\n"
    )


def _build_metrics_table(rows: list[dict[str, Any]]) -> str:
    header_cells = [
        "Adapter", "Recall@k", "p50 lat (ms)", "p95 lat (ms)", "QPS (est.)",
        "Ingest (vps)", "Index size", "Peak RSS",
    ]
    th_html = "".join(f"<th>{_esc(h)}</th>" for h in header_cells)

    body_rows = []
    for row in rows:
        label = _esc(row.get("label", row.get("db", "?")))
        recall = _fmt_float(row.get("recall_at_k_mean"), 3)
        p50 = _fmt_float(row.get("latency_ms_p50"), 2)
        p95 = _fmt_float(row.get("latency_ms_p95"), 2)
        qps = _fmt_float(row.get("qps_estimate"), 1)
        ingest = _fmt_float(row.get("ingest_throughput_vps"), 0)
        index_bytes = int(row.get("index_bytes", 0) or 0)
        peak_rss = int(row.get("peak_rss_bytes", 0) or 0)
        body_rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{_esc(recall)}</td>"
            f"<td>{_esc(p50)}</td>"
            f"<td>{_esc(p95)}</td>"
            f"<td>{_esc(qps)}</td>"
            f"<td>{_esc(ingest)}</td>"
            f"<td>{_esc(_fmt_bytes(index_bytes))}</td>"
            f"<td>{_esc(_fmt_bytes(peak_rss))}</td>"
            f"</tr>"
        )

    body_html = "\n".join(body_rows)
    return (
        "<h2>Headline Metrics</h2>\n"
        "<div class='card' style='overflow-x:auto'>\n"
        f"  <table><thead><tr>{th_html}</tr></thead>\n"
        f"  <tbody>{body_html}</tbody></table>\n"
        "</div>\n"
    )


def _build_per_adapter_tables(rows: list[dict[str, Any]]) -> str:
    detail_cols = [
        ("db", "DB"),
        ("label", "Label"),
        ("profile", "Profile"),
        ("n_passages", "Passages"),
        ("n_queries", "Queries"),
        ("dim", "Dim"),
        ("ingest_s", "Ingest (s)"),
        ("ingest_throughput_vps", "Ingest (vps)"),
        ("index_s", "Index (s)"),
        ("index_bytes", "Index bytes"),
        ("latency_ms_mean", "Latency mean (ms)"),
        ("latency_ms_p50", "p50 (ms)"),
        ("latency_ms_p95", "p95 (ms)"),
        ("latency_ms_p99", "p99 (ms)"),
        ("recall_at_k_mean", "Recall@k mean"),
        ("recall_at_k_p50", "Recall@k p50"),
        ("ndcg_at_k_mean", "NDCG@k mean"),
        ("qps_estimate", "QPS (est.)"),
        ("peak_rss_bytes", "Peak RSS"),
        ("adapter_memory_bytes", "Adapter memory"),
    ]

    sections = []
    for row in rows:
        label = _esc(row.get("label", row.get("db", "?")))
        cells = ""
        for col_key, col_label in detail_cols:
            val = row.get(col_key)
            if val is None:
                display = "N/A"
            elif col_key.endswith("_bytes"):
                display = _fmt_bytes(int(val))
            elif isinstance(val, float):
                display = _fmt_float(val, 4)
            else:
                display = str(val)
            cells += f"<tr><td style='font-weight:600'>{_esc(col_label)}</td><td>{_esc(display)}</td></tr>"

        sections.append(
            f"<h2>Detail: {label}</h2>\n"
            "<div class='card' style='overflow-x:auto'>\n"
            f"  <table><tbody>{cells}</tbody></table>\n"
            "</div>\n"
        )

    return "\n".join(sections)


def _build_charts_section(charts_dir: Path | None) -> str:
    if charts_dir is None:
        return ""
    png_files = sorted(charts_dir.glob("*.png"))
    if not png_files:
        return ""

    cards = []
    for png_path in png_files:
        try:
            data_uri = _embed_png_as_data_uri(png_path)
        except OSError:
            continue
        caption = _esc(png_path.stem.replace("_", " ").replace("-", " ").title())
        cards.append(
            "<div class='chart-card'>\n"
            f"  <img src='{data_uri}' alt='{_esc(png_path.stem)}' loading='lazy' />\n"
            f"  <div class='chart-caption'>{caption}</div>\n"
            "</div>"
        )

    if not cards:
        return ""

    cards_html = "\n".join(cards)
    return (
        "<h2>Charts</h2>\n"
        "<div class='charts-grid'>\n"
        f"{cards_html}\n"
        "</div>\n"
    )


def _build_footer(version: str) -> str:
    repo_url = "https://github.com/BishBish123/vector-db-bench"
    return (
        "<footer>\n"
        f"  Generated by <strong>vdbbench v{_esc(version)}</strong> &mdash; "
        f"<a href='{_esc(repo_url)}'>GitHub repository</a>\n"
        "</footer>\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_html_report(
    parquet_path: Path,
    manifest_path: Path,
    output_path: Path,
    charts_dir: Path | None = None,
) -> None:
    """Render a self-contained HTML report from bench outputs.

    Parameters
    ----------
    parquet_path:
        Path to ``summary.parquet`` produced by ``vdbbench bench``.
    manifest_path:
        Path to ``bench_manifest.json`` produced by ``vdbbench bench``.
    output_path:
        Destination for the generated ``.html`` file (parent dirs are created
        automatically).
    charts_dir:
        Optional directory containing ``.png`` chart files (e.g. those written
        by ``vdbbench plot``). Each PNG is embedded as a base64 ``data:`` URI
        so the resulting HTML is fully self-contained.
    """
    import pandas as pd  # noqa: PLC0415

    # --- Read inputs ---
    df = pd.read_parquet(parquet_path)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest: dict[str, Any] = json.loads(manifest_text)

    rows = df.to_dict(orient="records")
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    git_hash = _git_hash()
    vdbbench_version = manifest.get("vdbbench_version", "unknown")

    # --- Build sections ---
    header_html = _build_header(manifest, generated_at, git_hash)
    methodology_html = _build_methodology(manifest)
    metrics_html = _build_metrics_table(rows)
    charts_html = _build_charts_section(charts_dir)
    per_adapter_html = _build_per_adapter_tables(rows)
    footer_html = _build_footer(vdbbench_version)

    # --- Assemble ---
    page = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='UTF-8' />\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1' />\n"
        f"  <title>vdbbench Report — v{_esc(vdbbench_version)}</title>\n"
        f"  <style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{header_html}"
        "<div class='container'>\n"
        f"{methodology_html}"
        f"{metrics_html}"
        f"{charts_html}"
        f"{per_adapter_html}"
        f"{footer_html}"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
