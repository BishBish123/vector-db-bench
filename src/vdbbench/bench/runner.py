"""The bench runner.

Given an `EncodedBundle` (corpus + dense vectors) and a list of
`(adapter, params)` configurations, drive every adapter through the same
ingest -> index -> warm-up -> measured-search lifecycle and collect:

- Per-query latency (ms)
- Per-query top-k pids (so recall/NDCG are reproducible from the dump)
- Per-(db, params) aggregate stats: ingest throughput, index build time,
  index disk footprint, latency mean/p50/p95, recall@k, NDCG@k

Outputs are parquet so a reviewer can re-run the analysis without
re-running the bench. The harness is intentionally adapter-agnostic —
per-DB knob exploration lives in the configurations the caller passes.

Saved alongside `summary.parquet` and `timings.parquet` is a
`bench_manifest.json` (schema_version starts at 1) that binds the
parquet files to the encoded bundle they ran against, captures encoder
identity / adapter versions / host metadata, and stamps the bench
spec — everything a reviewer needs to verify what was actually run.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata as _import_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from vdbbench import __version__ as _vdbbench_version
from vdbbench.adapters.base import (
    IndexStats,
    IngestStats,
    OptionalAdapterUnavailableError,
    VectorStoreAdapter,
)
from vdbbench.embed.encoder import EncodedBundle
from vdbbench.metrics.retrieval import (
    aggregate,
    build_qrel_index,
    ndcg_at_k,
    recall_at_k,
)

# Bumped any time the manifest schema changes in a way readers care about.
BENCH_MANIFEST_SCHEMA_VERSION: int = 1

_PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    # name -> default warmup_queries and repeats. Profile only nudges the
    # defaults if the caller leaves them unset (sentinel value -1); explicit
    # caller values always win.
    "cold": {"warmup_queries": 0, "repeats": 1},
    "warm": {"warmup_queries": 10, "repeats": 1},
    "p99": {"warmup_queries": 50, "repeats": 5},
}

# Sentinel: caller didn't set the field, so the profile may pick a default.
_UNSET_INT = -1


@dataclass(frozen=True)
class BenchSpec:
    """One adapter + its tuning knobs + how many query repeats to time.

    `profile` is an opinionated shorthand over `(warmup_queries, repeats)`:

    * `"cold"` — measure first-query latency (warmup_queries=0, repeats=1).
    * `"warm"` — default; warm cache before timing (warmup_queries=10).
    * `"p99"` — exercise the tail (warmup_queries=50, repeats=5).

    Explicit `warmup_queries` / `repeats` always win over the profile.
    """

    adapter: VectorStoreAdapter
    params: dict[str, object] = field(default_factory=dict)
    k: int = 10
    warmup_queries: int = _UNSET_INT
    repeats: int = _UNSET_INT
    label: str | None = None  # human-readable; defaults to adapter.name + params hash
    profile: str = "warm"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.profile not in _PROFILE_DEFAULTS:
            raise ValueError(
                f"unknown profile {self.profile!r}; expected one of {sorted(_PROFILE_DEFAULTS)}"
            )

        # Resolve sentinel -> profile default. Done with object.__setattr__
        # because the dataclass is frozen; this only fires when callers
        # leave the field unset.
        defaults = _PROFILE_DEFAULTS[self.profile]
        if self.warmup_queries == _UNSET_INT:
            object.__setattr__(self, "warmup_queries", defaults["warmup_queries"])
        if self.repeats == _UNSET_INT:
            object.__setattr__(self, "repeats", defaults["repeats"])

        if self.warmup_queries < 0:
            raise ValueError("warmup_queries must be non-negative")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")

    def params_hash(self) -> str:
        digest = hashlib.blake2b(
            json.dumps(self.params, sort_keys=True, default=str).encode(), digest_size=8
        ).hexdigest()
        return digest

    def display_label(self) -> str:
        return self.label or f"{self.adapter.name}:{self.params_hash()}"


@dataclass(frozen=True)
class QueryTiming:
    """One row in the per-query results table."""

    db: str
    params_hash: str
    repeat: int
    qid: str
    latency_ms: float
    retrieved_pids: tuple[str, ...]


@dataclass(frozen=True)
class RunSummary:
    """One row in the per-(db, params) summary table.

    Memory columns:
      * ``baseline_rss_bytes`` — RSS of the bench process **before** any
        adapter work. Captures the constant Python interpreter / numpy /
        pandas footprint so it can be subtracted out.
      * ``index_rss_bytes`` — RSS after build_index returned (so embedded
        adapters' index footprint is included; service adapters report
        only the harness's own growth). Already baseline-subtracted —
        this is the adapter-attributable delta, not the raw RSS.
      * ``peak_rss_bytes`` — running max of the baseline-subtracted RSS
        across every phase (setup, ingest, build_index, warm-up,
        measured queries). The previous three-checkpoint version missed
        transient spikes between phases and reported the constant Python
        overhead alongside the adapter's own growth.
      * ``adapter_memory_bytes`` — what the adapter itself reports via
        ``memory_footprint_bytes()``. Some adapters return 0 (server-side
        memory is not meaningful per-table); kept here so a reviewer can
        compare vs. the harness RSS without touching adapter internals.
    """

    db: str
    label: str
    params_hash: str
    params_json: str
    profile: str
    n_passages: int
    n_queries: int
    dim: int
    ingest_s: float
    ingest_throughput_vps: float
    index_s: float
    index_bytes: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    recall_at_k_mean: float
    recall_at_k_p50: float
    ndcg_at_k_mean: float
    qps_estimate: float  # 1000 / latency_ms_mean
    baseline_rss_bytes: int = 0
    index_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    adapter_memory_bytes: int = 0


@dataclass(frozen=True)
class SkippedSpec:
    """One spec that bench skipped because the adapter wasn't reachable.

    Captured in ``BenchResult.skipped`` whenever ``run_bench`` is invoked
    with ``tolerate_failures=True``. The default (single-adapter) call
    path raises instead, so this only matters for ``--all`` runs.
    """

    label: str
    db: str
    reason: str
    error_type: str


@dataclass(frozen=True)
class BenchResult:
    """Per-query timings + per-(db, params) summary, both ready for parquet.

    `manifest` is a dict shaped like `bench_manifest.json` (see
    `_build_manifest()`); `save()` writes it next to the parquet files
    so a reviewer can verify which encoded bundle, encoder, and
    adapter versions produced the numbers.

    ``skipped`` is non-empty when a tolerant run (``--all`` mode) lost
    adapters to connection errors. The structured rows feed both the
    CLI summary line and the manifest, so the gap is auditable rather
    than silent.
    """

    timings: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object] = field(default_factory=dict)
    skipped: tuple[SkippedSpec, ...] = ()

    def save(self, root: str | Path) -> Path:
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        self.timings.to_parquet(out / "timings.parquet", index=False)
        self.summary.to_parquet(out / "summary.parquet", index=False)
        if self.manifest:
            (out / "bench_manifest.json").write_text(
                json.dumps(self.manifest, indent=2, sort_keys=True, default=str)
            )
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# Connection-class errors we recognise as "adapter unavailable" in
# tolerant (`--all`) mode. Anything outside this tuple still bubbles —
# it's a real bug, not a missing service. ``OSError`` covers
# ``ConnectionError``, ``ConnectionRefusedError``, and the network
# refused/timed-out cases the real adapter clients raise (psycopg's
# ``OperationalError`` and qdrant_client's transport errors both
# inherit from ``OSError`` via the underlying socket / requests stack).
_TOLERATED_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    OSError,
    OptionalAdapterUnavailableError,
)


def run_bench(
    encoded: EncodedBundle,
    specs: list[BenchSpec],
    *,
    progress: bool = False,
    tolerate_failures: bool = False,
) -> BenchResult:
    """Run every spec against `encoded` and return per-query + summary tables.

    Each spec is independently driven through the full lifecycle:
        adapter.setup(dim, params)
        adapter.ingest(pids, passage_vectors)
        adapter.build_index()
        # warm-up queries (timings discarded)
        # measured queries x repeats
        adapter.teardown()

    Failure semantics:

    * ``tolerate_failures=False`` (default) — any spec failure raises
      and aborts the whole run. Right for the single-adapter case where
      the user explicitly named the DB and a connection error means
      "fix your service".
    * ``tolerate_failures=True`` — used by ``--all`` mode. A connection
      / unavailable-import error on one spec logs a structured skip
      and the run continues with the remaining specs. Non-connection
      errors (e.g. an adapter bug) still raise — silencing those would
      hide real regressions. ``BenchResult.skipped`` carries the
      structured skip rows so the CLI can surface them and the
      manifest can record them.
    """
    pids: list[str] = encoded.bundle.passages["pid"].astype(str).tolist()
    qids: list[str] = encoded.bundle.queries["qid"].astype(str).tolist()
    qrel_index = build_qrel_index(encoded.bundle.qrels)

    timing_rows: list[QueryTiming] = []
    summary_rows: list[RunSummary] = []
    skipped: list[SkippedSpec] = []

    bench_started_at = _dt.datetime.now(_dt.UTC)
    for spec in specs:
        if progress:
            print(f"[bench] {spec.display_label()} — setup", flush=True)
        try:
            _run_one_spec(
                spec,
                encoded,
                pids,
                qids,
                qrel_index,
                timing_rows,
                summary_rows,
            )
        except _TOLERATED_FAILURE_TYPES as exc:
            if not tolerate_failures:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append(
                SkippedSpec(
                    label=spec.display_label(),
                    db=spec.adapter.name,
                    reason=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            # Structured single-line log — matches the `[bench]` prefix
            # of the progress line so the skip is grep-able alongside
            # the runs that succeeded.
            print(
                f"[bench] SKIP adapter={spec.adapter.name} "
                f"label={spec.display_label()} reason={reason}",
                flush=True,
            )
    bench_completed_at = _dt.datetime.now(_dt.UTC)

    manifest = _build_manifest(encoded, specs, bench_started_at, bench_completed_at)
    if skipped:
        manifest["skipped_specs"] = [asdict(s) for s in skipped]
    return BenchResult(
        timings=pd.DataFrame([_qt_to_row(t) for t in timing_rows]),
        summary=pd.DataFrame([asdict(r) for r in summary_rows]),
        manifest=manifest,
        skipped=tuple(skipped),
    )


def _run_one_spec(
    spec: BenchSpec,
    encoded: EncodedBundle,
    pids: list[str],
    qids: list[str],
    qrel_index: dict[str, dict[str, float]],
    timing_rows: list[QueryTiming],
    summary_rows: list[RunSummary],
) -> None:
    """Drive a single spec through the full lifecycle.

    Extracted so tolerant mode can wrap exactly the per-spec scope in
    try/except without nesting the loop body. Mutates the shared
    timing / summary lists in place — keeping the caller's pattern.
    """
    # Baseline RSS — the constant Python interpreter / numpy / pandas
    # footprint that's already loaded before any adapter work. Every
    # subsequent sample is reported as a delta against this so the
    # numbers in summary.parquet are "adapter-attributable RSS", not
    # "the whole bench process".
    baseline_rss = _sample_rss_bytes(force_gc=True)
    peak_tracker = _PeakRssTracker(baseline=baseline_rss)
    spec.adapter.setup(encoded.dim, spec.params)
    peak_tracker.observe()  # post-setup
    try:
        ingest_stats = spec.adapter.ingest(pids, encoded.passage_vectors)
        peak_tracker.observe()  # post-ingest
        index_stats = spec.adapter.build_index()
        index_rss_delta = peak_tracker.observe()  # post-index_build

        # Warm-up — exercise the cache + kick the JIT path before timing.
        warmup_n = min(spec.warmup_queries, len(qids))
        for i in range(warmup_n):
            spec.adapter.search(encoded.query_vectors[i], spec.k)
        peak_tracker.observe()  # post-warm-up

        # Measured runs.
        recalls: list[float] = []
        ndcgs: list[float] = []
        latencies_ms: list[float] = []
        for repeat in range(spec.repeats):
            for i, qid in enumerate(qids):
                qvec = encoded.query_vectors[i]
                t0 = time.perf_counter()
                retrieved = spec.adapter.search(qvec, spec.k)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                timing_rows.append(
                    QueryTiming(
                        db=spec.adapter.name,
                        params_hash=spec.params_hash(),
                        repeat=repeat,
                        qid=qid,
                        latency_ms=latency_ms,
                        retrieved_pids=tuple(retrieved),
                    )
                )
                latencies_ms.append(latency_ms)
                if qid in qrel_index:
                    rel = qrel_index[qid]
                    recalls.append(recall_at_k(retrieved, rel, spec.k))
                    ndcgs.append(ndcg_at_k(retrieved, rel, spec.k))

        peak_tracker.observe()  # post-measured-queries
        adapter_mem = 0
        try:
            adapter_mem = int(spec.adapter.memory_footprint_bytes())
        except Exception:  # pragma: no cover - adapter-level instability
            adapter_mem = 0

        summary_rows.append(
            _build_summary(
                spec,
                encoded,
                ingest_stats,
                index_stats,
                recalls,
                ndcgs,
                latencies_ms,
                baseline_rss=baseline_rss,
                index_rss=index_rss_delta,
                peak_rss=peak_tracker.peak,
                adapter_memory=adapter_mem,
            )
        )
    finally:
        spec.adapter.teardown()


class _PeakRssTracker:
    """Running max of baseline-subtracted RSS across pipeline phases.

    Three checkpoints (before-setup, after-index, after-queries) miss
    transient spikes — e.g. an HNSW build that frees pages before the
    next sample lands. ``observe()`` is called after each phase and
    returns the current delta; ``peak`` is the max across every call.
    Values are clamped at 0 so a transient drop below baseline (Python
    freeing pages mid-bench) doesn't show up as a negative footprint.
    """

    def __init__(self, *, baseline: int) -> None:
        self._baseline = baseline
        self._peak = 0

    def observe(self) -> int:
        """Take a sample, update the running peak, return the current delta."""
        sample = _sample_rss_bytes(force_gc=True)
        delta = max(sample - self._baseline, 0)
        self._peak = max(self._peak, delta)
        return delta

    @property
    def peak(self) -> int:
        return self._peak


def _build_manifest(
    encoded: EncodedBundle,
    specs: list[BenchSpec],
    started_at: _dt.datetime,
    completed_at: _dt.datetime,
) -> dict[str, object]:
    """Build the `bench_manifest.json` payload.

    Captures everything a reviewer needs to verify what was actually
    run: schema version (so future readers can branch), encoded bundle
    fingerprint, encoder identity, per-adapter package versions
    (best-effort via importlib.metadata), and best-effort host metadata
    (no PII — just CPU/OS/python info that frames the latency numbers).
    """
    return {
        "schema_version": BENCH_MANIFEST_SCHEMA_VERSION,
        "encoded_bundle_fingerprint": encoded.bundle.fingerprint(),
        "encoder_name": encoded.encoder_name,
        "encoder_dim": int(encoded.dim),
        "adapter_versions": _adapter_versions(specs),
        "host_metadata": _host_metadata(),
        "bench_started_at": started_at.isoformat(),
        "bench_completed_at": completed_at.isoformat(),
        "vdbbench_version": _vdbbench_version,
        "bench_specs": [_spec_to_dict(s) for s in specs],
    }


def _spec_to_dict(spec: BenchSpec) -> dict[str, object]:
    """A jsonable view of a BenchSpec — adapter as `name`, params verbatim."""
    return {
        "adapter": spec.adapter.name,
        "params": dict(spec.params),
        "k": int(spec.k),
        "warmup_queries": int(spec.warmup_queries),
        "repeats": int(spec.repeats),
        "label": spec.display_label(),
        "profile": spec.profile,
        "params_hash": spec.params_hash(),
    }


def _adapter_versions(specs: list[BenchSpec]) -> dict[str, str]:
    """Resolve installed package versions for every adapter in `specs`.

    Maps adapter.name -> "package==version". Unknown adapters and
    missing packages get the literal "unknown" so the manifest never
    misses a row, but a careful reviewer can spot the gap.
    """
    # adapter.name -> the pypi distribution that backs it.
    backends: dict[str, str] = {
        "pgvector": "pgvector",
        "qdrant": "qdrant-client",
        "lancedb": "lancedb",
        "chroma": "chromadb",
    }
    out: dict[str, str] = {}
    for spec in specs:
        name = spec.adapter.name
        if name in out:
            continue
        pkg = backends.get(name)
        if pkg is None:
            out[name] = "unknown"
            continue
        try:
            out[name] = f"{pkg}=={_import_metadata.version(pkg)}"
        except _import_metadata.PackageNotFoundError:
            out[name] = f"{pkg}==unknown"
    return out


def _host_metadata() -> dict[str, object]:
    """Best-effort (no-PII) machine metadata.

    Captures CPU / OS / interpreter so reviewers can frame the latency
    numbers — "this was a 4-core macOS Intel laptop, not an EC2 box"
    — without leaking the user's hostname, MAC, or working directory.
    Total memory comes from psutil if available; the bench already
    depends on psutil for memory sampling so this is free.
    """
    meta: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
    }
    try:
        import os  # noqa: PLC0415

        meta["cpu_count"] = os.cpu_count() or 0
    except Exception:  # pragma: no cover - defensive
        meta["cpu_count"] = 0
    try:
        import psutil  # noqa: PLC0415

        meta["total_memory_bytes"] = int(psutil.virtual_memory().total)
    except Exception:  # pragma: no cover - psutil missing or unreadable
        meta["total_memory_bytes"] = 0
    return meta


def _qt_to_row(qt: QueryTiming) -> dict[str, object]:
    # `retrieved_pids` is a tuple — parquet handles list-of-strings cleanly.
    return {
        "db": qt.db,
        "params_hash": qt.params_hash,
        "repeat": qt.repeat,
        "qid": qt.qid,
        "latency_ms": qt.latency_ms,
        "retrieved_pids": list(qt.retrieved_pids),
    }


def _sample_rss_bytes(*, force_gc: bool = False) -> int:
    """Best-effort RSS sample for the bench process.

    psutil is a required dep, but we keep the import lazy + defensive so
    a stripped install can't break the whole run. Returns 0 on failure
    so the column stays well-typed in the parquet output.

    ``force_gc=True`` runs ``gc.collect()`` before the sample. We use it
    on every checkpoint so deferred garbage doesn't show up as adapter
    growth in the running peak — slower than a raw sample, but the
    handful of collects per spec is invisible alongside ingest /
    index_build / query phases.
    """
    if force_gc:
        import gc  # noqa: PLC0415

        gc.collect()
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - psutil missing or unreadable
        return 0


def _build_summary(
    spec: BenchSpec,
    encoded: EncodedBundle,
    ingest: IngestStats,
    index: IndexStats,
    recalls: list[float],
    ndcgs: list[float],
    latencies_ms: list[float],
    *,
    baseline_rss: int = 0,
    index_rss: int = 0,
    peak_rss: int = 0,
    adapter_memory: int = 0,
) -> RunSummary:
    lat = aggregate(latencies_ms)
    rec = aggregate(recalls)
    ndcg = aggregate(ndcgs)
    qps = 1000.0 / lat["mean"] if lat["mean"] and not np.isnan(lat["mean"]) else 0.0
    p99 = (
        float(np.percentile(np.fromiter(latencies_ms, dtype=np.float64), 99))
        if latencies_ms
        else float("nan")
    )
    return RunSummary(
        db=spec.adapter.name,
        label=spec.display_label(),
        params_hash=spec.params_hash(),
        params_json=json.dumps(spec.params, sort_keys=True, default=str),
        profile=spec.profile,
        n_passages=encoded.bundle.n_passages,
        n_queries=encoded.bundle.n_queries,
        dim=encoded.dim,
        ingest_s=ingest.elapsed_s,
        ingest_throughput_vps=ingest.throughput_vps,
        index_s=index.elapsed_s,
        index_bytes=index.bytes_disk,
        latency_ms_mean=lat["mean"],
        latency_ms_p50=lat["p50"],
        latency_ms_p95=lat["p95"],
        latency_ms_p99=p99,
        recall_at_k_mean=rec["mean"],
        recall_at_k_p50=rec["p50"],
        ndcg_at_k_mean=ndcg["mean"],
        qps_estimate=qps,
        baseline_rss_bytes=int(baseline_rss),
        index_rss_bytes=int(index_rss),
        peak_rss_bytes=int(peak_rss),
        adapter_memory_bytes=int(adapter_memory),
    )
