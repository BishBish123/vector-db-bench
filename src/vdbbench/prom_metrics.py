"""Prometheus metrics for vdbbench bench runs.

Defines the metric objects that :func:`vdbbench.bench.runner.run_bench`
observes during a benchmark run. The module is safe to import even when no
Prometheus exporter is running — all metrics are collected in-process and
only *pushed* to a scrape endpoint when :func:`start_exporter` is called.

Metric inventory
----------------
INGEST_VECTORS_TOTAL
    Counter. Incremented by the number of vectors ingested per adapter run.
    Labels: ``adapter`` (adapter name).

QUERY_LATENCY_SECONDS
    Histogram. Records per-query latency in seconds.
    Buckets: [0.001, 0.005, 0.007, 0.01, 0.015, 0.025, 0.035, 0.05, 0.1, 0.25, 0.5, 1.0] s.
    Labels: ``adapter``.

QUERY_RECALL_AT_K
    Gauge. Per-(adapter, k) mean recall@k from the most recently completed
    spec. Updated once per spec, not once per query.
    Labels: ``adapter``, ``k``.

BENCH_DURATION_SECONDS
    Summary. Total wall-clock duration of a bench spec lifecycle
    (setup + ingest + build_index + warm-up + measured queries + teardown).
    Labels: ``adapter``.

Usage
-----
Import the module, call the observe helpers, and optionally start an HTTP
exporter on a port:

.. code-block:: python

    from vdbbench import prom_metrics

    prom_metrics.observe_ingest(adapter_name="qdrant", n_vectors=10_000)
    prom_metrics.observe_query_latency(adapter_name="qdrant", latency_s=0.012)
    prom_metrics.observe_recall(adapter_name="qdrant", k=10, recall=0.97)
    prom_metrics.observe_bench_duration(adapter_name="qdrant", duration_s=42.3)

    server = prom_metrics.start_exporter(port=9100)  # curl localhost:9100/metrics
    # ... do work ...
    prom_metrics.stop_exporter(server)       # port released
"""

from __future__ import annotations

from wsgiref.simple_server import WSGIServer

import prometheus_client as _prom

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

INGEST_VECTORS_TOTAL: _prom.Counter = _prom.Counter(
    "vdbbench_ingest_vectors_total",
    "Total number of vectors ingested across all bench specs.",
    labelnames=["adapter"],
)

QUERY_LATENCY_SECONDS: _prom.Histogram = _prom.Histogram(
    "vdbbench_query_latency_seconds",
    "Per-query latency observed during the measured bench phase.",
    labelnames=["adapter"],
    buckets=[0.001, 0.005, 0.007, 0.01, 0.015, 0.025, 0.035, 0.05, 0.1, 0.25, 0.5, 1.0],
)

QUERY_RECALL_AT_K: _prom.Gauge = _prom.Gauge(
    "vdbbench_query_recall_at_k",
    "Mean recall@k from the most recently completed spec for this adapter.",
    labelnames=["adapter", "k"],
)

BENCH_DURATION_SECONDS: _prom.Summary = _prom.Summary(
    "vdbbench_bench_duration_seconds",
    "Wall-clock duration of a full bench spec lifecycle "
    "(setup + ingest + index + warm-up + queries + teardown).",
    labelnames=["adapter"],
)

# ---------------------------------------------------------------------------
# Observe helpers
# ---------------------------------------------------------------------------


def observe_ingest(*, adapter_name: str, n_vectors: int) -> None:
    """Increment :data:`INGEST_VECTORS_TOTAL` by ``n_vectors``."""
    INGEST_VECTORS_TOTAL.labels(adapter=adapter_name).inc(n_vectors)


def observe_query_latency(*, adapter_name: str, latency_s: float) -> None:
    """Record one query latency observation in :data:`QUERY_LATENCY_SECONDS`."""
    QUERY_LATENCY_SECONDS.labels(adapter=adapter_name).observe(latency_s)


def observe_recall(*, adapter_name: str, k: int, recall: float) -> None:
    """Set :data:`QUERY_RECALL_AT_K` for this adapter + k."""
    QUERY_RECALL_AT_K.labels(adapter=adapter_name, k=str(k)).set(recall)


def observe_bench_duration(*, adapter_name: str, duration_s: float) -> None:
    """Record one spec's total wall-clock duration in :data:`BENCH_DURATION_SECONDS`."""
    BENCH_DURATION_SECONDS.labels(adapter=adapter_name).observe(duration_s)


# ---------------------------------------------------------------------------
# HTTP exporter
# ---------------------------------------------------------------------------


def start_exporter(port: int) -> WSGIServer:
    """Start the Prometheus HTTP scrape endpoint on ``port``.

    Blocks until the background thread is up. After this call,
    ``curl localhost:<port>/metrics`` returns the current metric state.

    Returns the underlying :class:`wsgiref.simple_server.WSGIServer` so
    callers can stop the exporter with :func:`stop_exporter` (or by calling
    ``server.shutdown()`` directly).

    Calling this more than once with the same port raises
    ``OSError: [Errno 98] Address already in use`` from the underlying
    ``socketserver.TCPServer`` — it is the caller's responsibility to
    call this at most once per process, or to call :func:`stop_exporter`
    between restarts.

    prometheus_client >= 0.20 returns ``(server, thread)`` from
    ``start_http_server``; this function captures and returns the server.
    """
    server, _thread = _prom.start_http_server(port)
    return server


def stop_exporter(server: WSGIServer) -> None:
    """Shut down a Prometheus HTTP server returned by :func:`start_exporter`.

    Calls ``server.shutdown()`` which signals the background daemon thread
    to stop serving and blocks until it has exited. After this call the port
    is released and is no longer listening.

    It is safe to call this function more than once on the same server
    object — the second call is a no-op from the underlying
    ``socketserver.BaseServer`` perspective.
    """
    server.shutdown()
    server.server_close()
