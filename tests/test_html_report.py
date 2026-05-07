"""Tests for the HTML report renderer (vdbbench.report.html)."""

from __future__ import annotations

import html.parser
import json
import struct
import zlib
from pathlib import Path

import pandas as pd
import pytest

from vdbbench.report.html import render_html_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SUMMARY_ROWS = [
    {
        "db": "pgvector",
        "label": "pgvector:hnsw-default",
        "params_hash": "abc123",
        "params_json": '{"index": "hnsw", "metric": "cosine"}',
        "profile": "warm",
        "n_passages": 5000,
        "n_queries": 100,
        "dim": 64,
        "ingest_s": 2.5,
        "ingest_throughput_vps": 2000.0,
        "index_s": 4.5,
        "index_bytes": 4661248,
        "latency_ms_mean": 11.1,
        "latency_ms_p50": 9.8,
        "latency_ms_p95": 19.0,
        "latency_ms_p99": 30.4,
        "recall_at_k_mean": 0.88,
        "recall_at_k_p50": 1.0,
        "ndcg_at_k_mean": 0.926,
        "qps_estimate": 89.8,
        "baseline_rss_bytes": 184508416,
        "index_rss_bytes": 2105344,
        "peak_rss_bytes": 2121728,
        "adapter_memory_bytes": 0,
    },
    {
        "db": "qdrant",
        "label": "qdrant:hnsw-default",
        "params_hash": "def456",
        "params_json": '{"metric": "cosine"}',
        "profile": "warm",
        "n_passages": 5000,
        "n_queries": 100,
        "dim": 64,
        "ingest_s": 2.3,
        "ingest_throughput_vps": 2155.0,
        "index_s": 0.0,
        "index_bytes": 0,
        "latency_ms_mean": 19.9,
        "latency_ms_p50": 18.4,
        "latency_ms_p95": 33.9,
        "latency_ms_p99": 40.3,
        "recall_at_k_mean": 1.0,
        "recall_at_k_p50": 1.0,
        "ndcg_at_k_mean": 1.0,
        "qps_estimate": 50.2,
        "baseline_rss_bytes": 191041536,
        "index_rss_bytes": 15675392,
        "peak_rss_bytes": 15736832,
        "adapter_memory_bytes": 0,
    },
]

_MANIFEST = {
    "schema_version": 1,
    "vdbbench_version": "0.1.0",
    "encoder_name": "synthetic-gaussian",
    "encoder_dim": 64,
    "bench_started_at": "2026-05-06T05:05:42.504524+00:00",
    "bench_completed_at": "2026-05-06T05:06:00.472289+00:00",
    "partial": False,
    "adapter_versions": {
        "pgvector": "pgvector==0.4.2",
        "qdrant": "qdrant-client==1.17.1",
    },
    "bench_specs": [
        {
            "adapter": "pgvector",
            "label": "pgvector:hnsw-default",
            "k": 10,
            "profile": "warm",
            "repeats": 1,
            "warmup_queries": 10,
            "params": {"index": "hnsw", "metric": "cosine"},
            "params_hash": "abc123",
        },
        {
            "adapter": "qdrant",
            "label": "qdrant:hnsw-default",
            "k": 10,
            "profile": "warm",
            "repeats": 1,
            "warmup_queries": 10,
            "params": {"metric": "cosine"},
            "params_hash": "def456",
        },
    ],
    "host_metadata": {
        "cpu_count": 8,
        "machine": "x86_64",
        "platform": "macOS-15.7.4-x86_64-i386-64bit",
        "processor": "i386",
        "python_implementation": "CPython",
        "python_version": "3.12.13",
        "total_memory_bytes": 8589934592,
    },
    "encoded_bundle_fingerprint": "aabbcc",
}


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "summary.parquet"
    pd.DataFrame(_SUMMARY_ROWS).to_parquet(p, index=False)
    return p


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    p = tmp_path / "bench_manifest.json"
    p.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    return p


@pytest.fixture()
def charts_dir(tmp_path: Path) -> Path:
    """Create tiny 1x1 PNG files in a charts directory (no PIL dep)."""

    def _make_png_1x1() -> bytes:
        # 1x1 RGB red pixel: filter byte 0 + R=255 G=0 B=0
        raw = b"\x00\xff\x00\x00"
        compressed = zlib.compress(raw)

        def chunk(tag: bytes, data: bytes) -> bytes:
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        return sig + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")

    d = tmp_path / "charts"
    d.mkdir()
    png_bytes = _make_png_1x1()
    (d / "pareto.png").write_bytes(png_bytes)
    (d / "recall.png").write_bytes(png_bytes)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderHtmlReport:
    def test_produces_nonempty_html(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 500

    def test_html_starts_with_doctype(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")

    def test_html_is_parseable(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        # html.parser.HTMLParser raises no exception on valid HTML; we just check
        # that parsing completes without raising.
        parser = html.parser.HTMLParser()
        parser.feed(content)  # raises if malformed in strict mode

    def test_methodology_section_present(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "Methodology" in content

    def test_metrics_section_present(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "Headline Metrics" in content

    def test_adapter_labels_in_output(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "pgvector:hnsw-default" in content
        assert "qdrant:hnsw-default" in content

    def test_charts_embedded_as_data_uri(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path, charts_dir: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
            charts_dir=charts_dir,
        )
        content = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_charts_section_present_when_charts_dir_given(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path, charts_dir: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
            charts_dir=charts_dir,
        )
        content = out.read_text(encoding="utf-8")
        assert "Charts" in content

    def test_charts_section_absent_when_no_charts_dir(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
            charts_dir=None,
        )
        content = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," not in content

    def test_footer_has_version(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "0.1.0" in content  # vdbbench_version from manifest
        assert "github.com" in content

    def test_host_info_in_methodology(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        # RAM: 8 GB
        assert "8.0 GB" in content
        # CPU count
        assert "8" in content

    def test_creates_parent_directories(
        self, parquet_path: Path, manifest_path: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "deeply" / "nested" / "report.html"
        render_html_report(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            output_path=out,
        )
        assert out.exists()
