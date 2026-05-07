"""Tests for vdbbench.pricing and vdbbench.prom_metrics.

Covers:
- pricing table structure and cost formula
- cost_per_million_queries_usd edge cases
- prom_metrics module imports and metric increments
- run_bench emits histogram observations (counter check)
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pandas as pd
import pytest

import vdbbench.prom_metrics as pm
from vdbbench.adapters.base import IndexStats, IngestStats
from vdbbench.bench.runner import BenchSpec, run_bench
from vdbbench.corpus.bundle import CorpusBundle
from vdbbench.embed.encoder import EncodedBundle, FakeEncoder, encode_corpus
from vdbbench.pricing import PRICING_TABLE, AdapterPricing, cost_per_million_queries_usd

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _toy_encoded() -> EncodedBundle:
    bundle = CorpusBundle(
        name="toy",
        passages=pd.DataFrame(
            {"pid": [f"p{i}" for i in range(8)], "text": [f"doc {i}" for i in range(8)]}
        ),
        queries=pd.DataFrame({"qid": ["q0", "q1"], "text": ["doc 0", "doc 3"]}),
        qrels=pd.DataFrame({"qid": ["q0", "q1"], "pid": ["p0", "p3"], "relevance": [1, 1]}),
    )
    return encode_corpus(bundle, FakeEncoder(dim=16))


class _MemAdapter:
    name = "exact"

    def __init__(self, search_sleep_ms: float = 0.0) -> None:
        self._dim: int | None = None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._search_sleep_ms = search_sleep_ms

    def setup(self, dim: int, params: dict[str, object]) -> None:
        self._dim = dim
        self._ids = []
        self._mat = None

    def teardown(self) -> None:
        self._ids = []
        self._mat = None
        self._dim = None

    def ingest(self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024) -> IngestStats:
        self._ids = list(ids)
        self._mat = vectors.astype(np.float32, copy=False)
        return IngestStats(n_vectors=len(ids), elapsed_s=0.0)

    def build_index(self) -> IndexStats:
        return IndexStats(elapsed_s=0.0, bytes_disk=0)

    def search(self, query: np.ndarray, k: int) -> list[str]:
        if self._search_sleep_ms:
            time.sleep(self._search_sleep_ms / 1000.0)
        assert self._mat is not None
        sims = self._mat @ query
        top_idx = np.argsort(-sims)[:k]
        return [self._ids[i] for i in top_idx]

    def memory_footprint_bytes(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Pricing table tests
# ---------------------------------------------------------------------------


class TestPricingTable:
    def test_all_entries_are_adapter_pricing(self) -> None:
        for key, entry in PRICING_TABLE.items():
            assert isinstance(entry, AdapterPricing), f"{key} is not AdapterPricing"

    def test_known_adapters_present(self) -> None:
        for name in ("pgvector", "qdrant", "chroma", "lancedb", "exact"):
            assert name in PRICING_TABLE, f"{name} missing from PRICING_TABLE"

    def test_pgvector_verified(self) -> None:
        assert PRICING_TABLE["pgvector"].verified is True
        assert PRICING_TABLE["pgvector"].compute_per_hour_usd == pytest.approx(0.222)

    def test_exact_is_zero_cost(self) -> None:
        p = PRICING_TABLE["exact"]
        assert p.compute_per_hour_usd == 0.0
        assert p.per_query_usd == 0.0

    def test_unverified_entries_have_note(self) -> None:
        for key, entry in PRICING_TABLE.items():
            if not entry.verified:
                assert entry.notes, f"{key} is unverified but has no notes"
                assert "UNVERIFIED" in entry.notes or "unverified" in entry.notes.lower(), (
                    f"{key} unverified entry notes do not mention UNVERIFIED"
                )

    def test_chroma_per_query_nonzero(self) -> None:
        # Chroma Cloud charges per TiB queried, so per_query_usd > 0.
        assert PRICING_TABLE["chroma"].per_query_usd > 0


# ---------------------------------------------------------------------------
# cost_per_million_queries_usd formula tests
# ---------------------------------------------------------------------------


class TestCostFormula:
    def test_exact_adapter_returns_none(self) -> None:
        # exact is a non-cloud in-process baseline; returning 0.0 would be
        # misleading ("cheapest cloud option") so the function returns None
        # (callers convert to NaN in parquet).
        result = cost_per_million_queries_usd("exact", total_queries=1000, total_query_seconds=1.0)
        assert result is None

    def test_pgvector_reasonable_magnitude(self) -> None:
        # 1000 QPS -> $0.222 / (1000*3600) per query -> ~$0.0617/M
        result = cost_per_million_queries_usd(
            "pgvector", total_queries=1000, total_query_seconds=1.0
        )
        assert result is not None
        assert 0.001 < result < 10.0, f"unreasonable $/M cost: {result}"

    def test_unknown_adapter_returns_none(self) -> None:
        result = cost_per_million_queries_usd(
            "nonexistent_db", total_queries=100, total_query_seconds=1.0
        )
        assert result is None

    def test_zero_queries_returns_none(self) -> None:
        result = cost_per_million_queries_usd("pgvector", total_queries=0, total_query_seconds=1.0)
        assert result is None

    def test_zero_seconds_returns_none(self) -> None:
        result = cost_per_million_queries_usd(
            "pgvector", total_queries=100, total_query_seconds=0.0
        )
        assert result is None

    def test_formula_correctness(self) -> None:
        # Hand-calculate for pgvector:
        # qps = 100/1.0 = 100; hourly = 360000
        # cost_per_q = 0.222/360000 + 0 = 6.1667e-7
        # cost_per_M = 6.1667e-7 * 1e6 = 0.6167
        result = cost_per_million_queries_usd(
            "pgvector", total_queries=100, total_query_seconds=1.0
        )
        expected = (0.222 / (100 * 3600)) * 1_000_000
        assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Helpers for reading prometheus_client metric internals
# ---------------------------------------------------------------------------


def _histogram_count(metric_family: object, adapter: str) -> float:
    """Extract the _count sample from a labeled Histogram via collect()."""
    for mf in metric_family.collect():  # type: ignore[union-attr]
        for sample in mf.samples:
            if sample.name.endswith("_count") and sample.labels.get("adapter") == adapter:
                return float(sample.value)
    return 0.0


def _summary_count(labeled_summary: object) -> float:
    """Extract observation count from a labeled Summary via _count attribute."""
    return float(labeled_summary._count.get())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# prom_metrics module tests
# ---------------------------------------------------------------------------


class TestPromMetrics:
    def test_module_importable(self) -> None:
        assert hasattr(pm, "INGEST_VECTORS_TOTAL")
        assert hasattr(pm, "QUERY_LATENCY_SECONDS")
        assert hasattr(pm, "QUERY_RECALL_AT_K")
        assert hasattr(pm, "BENCH_DURATION_SECONDS")

    def test_observe_ingest_increments(self) -> None:
        # Counter exposes ._value internally.
        before = pm.INGEST_VECTORS_TOTAL.labels(adapter="_test_ingest")._value.get()
        pm.observe_ingest(adapter_name="_test_ingest", n_vectors=42)
        after = pm.INGEST_VECTORS_TOTAL.labels(adapter="_test_ingest")._value.get()
        assert after - before == pytest.approx(42)

    def test_observe_query_latency_increments_count(self) -> None:
        # Histogram doesn't expose ._count directly; use collect() to read the
        # _count sample from the metric family.
        before = _histogram_count(pm.QUERY_LATENCY_SECONDS, "_test_lat2")
        pm.observe_query_latency(adapter_name="_test_lat2", latency_s=0.01)
        pm.observe_query_latency(adapter_name="_test_lat2", latency_s=0.02)
        after = _histogram_count(pm.QUERY_LATENCY_SECONDS, "_test_lat2")
        assert after - before == pytest.approx(2)

    def test_observe_recall_sets_gauge(self) -> None:
        pm.observe_recall(adapter_name="_test_rec", k=10, recall=0.95)
        val = pm.QUERY_RECALL_AT_K.labels(adapter="_test_rec", k="10")._value.get()
        assert val == pytest.approx(0.95)

    def test_observe_bench_duration_increments_count(self) -> None:
        labeled = pm.BENCH_DURATION_SECONDS.labels(adapter="_test_dur")
        before = _summary_count(labeled)
        pm.observe_bench_duration(adapter_name="_test_dur", duration_s=5.0)
        after = _summary_count(pm.BENCH_DURATION_SECONDS.labels(adapter="_test_dur"))
        assert after - before == pytest.approx(1)


# ---------------------------------------------------------------------------
# run_bench integration: histogram observe count increases
# ---------------------------------------------------------------------------


class TestRunBenchEmitsMetrics:
    def test_query_latency_histogram_increments_after_run(self) -> None:
        """run_bench with the exact adapter must increase the histogram count."""
        adapter_name = "exact"
        before = _histogram_count(pm.QUERY_LATENCY_SECONDS, adapter_name)

        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=3, warmup_queries=0, repeats=1)
        result = run_bench(encoded, [spec])

        after = _histogram_count(pm.QUERY_LATENCY_SECONDS, adapter_name)
        # 2 queries x 1 repeat = 2 latency observations
        assert after - before == pytest.approx(len(result.timings))

    def test_summary_has_cost_column(self) -> None:
        """summary.parquet must include cost_per_million_queries_usd."""
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=3, warmup_queries=0, repeats=1)
        result = run_bench(encoded, [spec])
        assert "cost_per_million_queries_usd" in result.summary.columns

    def test_exact_adapter_cost_is_nan(self) -> None:
        """exact adapter should produce NaN (not 0.0) for cost — it is a
        non-cloud in-process baseline; 0.0 would read as "cheapest cloud
        option" which is semantically wrong. NaN signals "not applicable"."""
        import math  # noqa: PLC0415

        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=3, warmup_queries=0, repeats=1)
        result = run_bench(encoded, [spec])
        cost = result.summary.iloc[0]["cost_per_million_queries_usd"]
        assert math.isnan(float(cost)), f"expected NaN for exact adapter cost, got {cost}"


# ---------------------------------------------------------------------------
# Exporter lifecycle: start → scrape → stop → port released
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign an ephemeral port, then return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_is_listening(port: int) -> bool:
    """Return True if something is accepting TCP connections on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class TestExporterLifecycle:
    """start_exporter returns a server handle; stop_exporter releases the port."""

    def test_start_returns_server_handle(self) -> None:
        from wsgiref.simple_server import WSGIServer  # noqa: PLC0415

        port = _find_free_port()
        server = pm.start_exporter(port)
        try:
            assert isinstance(server, WSGIServer), (
                f"start_exporter must return WSGIServer, got {type(server)}"
            )
        finally:
            pm.stop_exporter(server)

    def test_metrics_endpoint_reachable_after_start(self) -> None:
        """After start_exporter the /metrics path must return a 200 response."""
        import urllib.request  # noqa: PLC0415

        port = _find_free_port()
        server = pm.start_exporter(port)
        try:
            url = f"http://127.0.0.1:{port}/metrics"
            with urllib.request.urlopen(url, timeout=3) as resp:
                assert resp.status == 200, f"expected 200 from {url}, got {resp.status}"
                body = resp.read().decode()
            # The default prometheus_client registry always exposes process metrics.
            assert "python_info" in body or "process_" in body, (
                "expected at least one default prometheus metric in /metrics body"
            )
        finally:
            pm.stop_exporter(server)

    def test_port_released_after_stop(self) -> None:
        """After stop_exporter the TCP port must no longer accept connections."""
        port = _find_free_port()
        server = pm.start_exporter(port)
        assert _port_is_listening(port), "port should be open after start_exporter"
        pm.stop_exporter(server)
        # Give the OS a moment to reclaim the port (shutdown() is synchronous,
        # but the kernel's TIME_WAIT state can linger briefly on some platforms).
        deadline = time.monotonic() + 2.0
        released = False
        while time.monotonic() < deadline:
            if not _port_is_listening(port):
                released = True
                break
            time.sleep(0.05)
        assert released, f"port {port} is still listening after stop_exporter"
