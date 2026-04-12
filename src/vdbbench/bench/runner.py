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

import contextlib
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
import uuid
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


class IncompatibleBenchManifestError(ValueError):
    """Raised by :func:`load_bench_manifest` when a manifest's schema_version
    doesn't match :data:`BENCH_MANIFEST_SCHEMA_VERSION`.

    The compatibility contract is one-way today (no backwards-compat
    shims yet), so a mismatch means the reader and writer disagree on
    layout and continuing would silently corrupt downstream analysis.
    The exception carries both versions so a caller can decide whether
    to upgrade vdbbench or pin to the producer's release.
    """

    def __init__(self, found: object, expected: int) -> None:
        super().__init__(
            f"bench_manifest.json schema_version {found!r} is not compatible "
            f"with this vdbbench (expected {expected})"
        )
        self.found = found
        self.expected = expected


def load_bench_manifest(path: str | Path) -> dict[str, object]:
    """Read a `bench_manifest.json` and validate its schema_version.

    Pairs with the writer in :class:`BenchResult.save`. The manifest's
    ``schema_version`` field has to be present and equal to
    :data:`BENCH_MANIFEST_SCHEMA_VERSION`; otherwise we raise
    :class:`IncompatibleBenchManifestError` so the contract isn't
    write-only.
    """
    raw = Path(path).read_text()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise IncompatibleBenchManifestError(type(payload).__name__, BENCH_MANIFEST_SCHEMA_VERSION)
    if "schema_version" not in payload:
        raise IncompatibleBenchManifestError(None, BENCH_MANIFEST_SCHEMA_VERSION)
    found = payload["schema_version"]
    if found != BENCH_MANIFEST_SCHEMA_VERSION:
        raise IncompatibleBenchManifestError(found, BENCH_MANIFEST_SCHEMA_VERSION)
    return payload

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


# Postgres caps identifiers at NAMEDATALEN-1 = 63 bytes; mirrored from
# vdbbench.adapters.pgvector to keep the runner self-contained without a
# circular import. Both ends enforce the same limit; the duplication is
# documented at both sites.
_PG_IDENT_MAX_BYTES = 63

# Hex chars from uuid4 to use as the per-run suffix. 16 hex = 64 bits of
# entropy, the upper half of a uuid4 — the birthday-collision probability
# over the lifetime of any realistic concurrent-run scenario is
# negligible. The previous 8 hex (32 bits) gave only ~10% collision
# probability after ~100k concurrent runs — low, but not the "no
# collisions" guarantee per-run isolation needs.
_TABLE_NAME_SUFFIX_HEX = 16


def _generate_table_name(prefix: str = "vdbbench_vectors") -> str:
    """Return ``<prefix>_<16hex>`` for one bench-run's table isolation.

    Two parallel ``run_bench`` calls against the same Postgres can't
    share a table, so each run gets a uuid-suffixed name. The full
    identifier is capped at Postgres' NAMEDATALEN-1 (63 bytes) — if the
    caller passes an unusually long prefix, we trim the prefix (not the
    suffix) so the entropy that protects the race is preserved.
    """
    suffix = uuid.uuid4().hex[:_TABLE_NAME_SUFFIX_HEX]
    # +1 for the underscore separator between prefix and suffix.
    max_prefix_bytes = _PG_IDENT_MAX_BYTES - len(suffix) - 1
    if max_prefix_bytes <= 0:
        # Should never happen with a sane suffix size; if a future
        # change bumps the suffix past 62 bytes, fall back to
        # suffix-only with a leading "v" so the result is still a legal
        # identifier (must start with a letter or underscore).
        return f"v{suffix}"[:_PG_IDENT_MAX_BYTES]
    if len(prefix.encode("utf-8")) > max_prefix_bytes:
        # uuid4 hex is ASCII so a byte-length truncate by chars is
        # exact for the suffix; only the *prefix* could be multibyte
        # (UTF-8 caller-supplied), so re-check after the char-trim.
        prefix = prefix[:max_prefix_bytes]
        while len(prefix.encode("utf-8")) > max_prefix_bytes:
            prefix = prefix[:-1]
    return f"{prefix}_{suffix}"


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
    # Per-run table / collection name. ``None`` => the runner generates a
    # uuid-suffixed name (``vdbbench_vectors_<8hex>``) so two concurrent
    # ``run_bench`` calls against the same Postgres can't share a table
    # and silently corrupt each other's results. Pass an explicit string
    # to pin the name (e.g. for repro of an earlier run).
    table_name: str | None = None

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

    ``partial`` is ``True`` when the result was interrupted by an
    exception mid-run and only contains results for the specs that
    completed before the failure. The manifest also carries this flag.
    """

    timings: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object] = field(default_factory=dict)
    skipped: tuple[SkippedSpec, ...] = ()
    partial: bool = False

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
# it's a real bug, not a missing service.
#
# ``OSError`` covers stdlib ``ConnectionError`` /
# ``ConnectionRefusedError`` and any raw socket-level failure surfaced by
# adapters that don't wrap their transport.
#
# Real adapter clients do NOT funnel through ``OSError``; their MROs are:
#   * ``psycopg.OperationalError`` ->
#         psycopg.DatabaseError -> psycopg.Error -> Exception
#   * ``qdrant_client.http.exceptions.ResponseHandlingException`` ->
#         qdrant_client.http.exceptions.ApiException -> Exception
#     (this is the wrapper the qdrant client raises on
#      ``httpx.ConnectError`` / ``httpx.ReadTimeout`` etc.)
#   * ``httpx.ConnectError`` ->
#         httpx.NetworkError -> httpx.TransportError ->
#         httpx.RequestError -> httpx.HTTPError -> Exception
#
# So each of those classes has to be added explicitly. We import them
# lazily — psycopg / qdrant_client / httpx are optional adapter deps,
# and a stripped install (e.g. tests against the in-memory adapter
# only) must still import this module.
def _resolve_tolerated_failure_types() -> tuple[type[BaseException], ...]:
    """Build the tolerated-failure tuple at import time.

    Each external client lookup is wrapped in a ``try`` so a missing
    package doesn't break the runner — the matching adapter would
    already raise ``OptionalAdapterUnavailableError`` (which IS in the
    tuple) before any of these classes could be the failure mode.
    """
    types: list[type[BaseException]] = [OSError, OptionalAdapterUnavailableError]
    try:
        import psycopg  # noqa: PLC0415

        types.append(psycopg.OperationalError)
    except ImportError:  # pragma: no cover - psycopg is a hard dep today
        pass
    try:
        from qdrant_client.http.exceptions import (  # noqa: PLC0415
            ApiException,
            ResponseHandlingException,
        )

        # ResponseHandlingException is what the client raises on
        # connect-time httpx errors; ApiException is the broader parent
        # so a future client release that bypasses the wrapper still
        # lands in the tolerated set.
        types.extend([ResponseHandlingException, ApiException])
    except ImportError:  # pragma: no cover - qdrant_client is a hard dep today
        pass
    try:
        import httpx  # noqa: PLC0415

        # httpx.TransportError covers ConnectError, ReadError, ConnectTimeout,
        # ReadTimeout, etc. — every "the service didn't respond" case.
        types.append(httpx.TransportError)
    except ImportError:  # pragma: no cover - httpx ships with qdrant-client
        pass
    return tuple(types)


_TOLERATED_FAILURE_TYPES: tuple[type[BaseException], ...] = _resolve_tolerated_failure_types()


def run_bench(
    encoded: EncodedBundle,
    specs: list[BenchSpec],
    *,
    progress: bool = False,
    tolerate_failures: bool = False,
    out: str | Path | None = None,
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

    Incremental writes (``out`` is not ``None``):

    When ``out`` is supplied, results are written to disk after every
    successfully completed spec so a mid-run crash doesn't discard all
    work. On an unhandled exception the completed specs (0..N-1) are
    persisted with ``partial=True`` in both the ``BenchResult`` and the
    ``bench_manifest.json`` before the exception is re-raised. The
    caller can therefore inspect ``out/`` to see how far the run got.
    """
    pids: list[str] = encoded.bundle.passages["pid"].astype(str).tolist()
    qids: list[str] = encoded.bundle.queries["qid"].astype(str).tolist()
    qrel_index = build_qrel_index(encoded.bundle.qrels)

    timing_rows: list[QueryTiming] = []
    summary_rows: list[RunSummary] = []
    skipped: list[SkippedSpec] = []

    out_path: Path | None = Path(out) if out is not None else None

    bench_started_at = _dt.datetime.now(_dt.UTC)
    try:
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
            # Write incremental results after each successful spec so a later
            # crash doesn't discard everything that already ran.
            if out_path is not None and summary_rows:
                _write_incremental(
                    out_path,
                    timing_rows,
                    summary_rows,
                    skipped,
                    encoded,
                    specs,
                    bench_started_at,
                    partial=False,
                )
    except BaseException:
        # On any unhandled exception OR Ctrl-C: persist the specs that already
        # completed (timing_rows / summary_rows accumulated so far) with
        # partial=True, then re-raise so the caller still sees the failure.
        # We deliberately catch BaseException — not Exception — so that
        # KeyboardInterrupt and SystemExit are still treated as bench
        # interruptions: a 30-minute run cancelled with Ctrl-C should leave
        # behind a partial manifest of the specs that already succeeded,
        # not silently throw all of that work away. The bare `raise` below
        # re-propagates the original exception (BaseException or otherwise)
        # so the interpreter still terminates / KeyboardInterrupt still
        # tears down the CLI as before.
        if out_path is not None:
            with contextlib.suppress(Exception):
                # Crash inside the manifest writer must not mask the
                # original interruption — swallow only its exceptions.
                _write_incremental(
                    out_path,
                    timing_rows,
                    summary_rows,
                    skipped,
                    encoded,
                    specs,
                    bench_started_at,
                    partial=True,
                )
        raise
    bench_completed_at = _dt.datetime.now(_dt.UTC)

    manifest = _build_manifest(encoded, specs, bench_started_at, bench_completed_at, partial=False)
    if skipped:
        manifest["skipped_specs"] = [asdict(s) for s in skipped]
    return BenchResult(
        timings=pd.DataFrame([_qt_to_row(t) for t in timing_rows]),
        summary=pd.DataFrame([asdict(r) for r in summary_rows]),
        manifest=manifest,
        skipped=tuple(skipped),
        partial=False,
    )


def _write_incremental(
    out_path: Path,
    timing_rows: list[QueryTiming],
    summary_rows: list[RunSummary],
    skipped: list[SkippedSpec],
    encoded: EncodedBundle,
    specs: list[BenchSpec],
    bench_started_at: _dt.datetime,
    *,
    partial: bool,
) -> None:
    """Flush timing/summary/manifest to ``out_path``.

    Called after each completed spec (``partial=False``) and once on
    exception (``partial=True``) so data is never fully lost to a crash.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    timings_df = pd.DataFrame([_qt_to_row(t) for t in timing_rows])
    summary_df = pd.DataFrame([asdict(r) for r in summary_rows])
    timings_df.to_parquet(out_path / "timings.parquet", index=False)
    summary_df.to_parquet(out_path / "summary.parquet", index=False)
    now = _dt.datetime.now(_dt.UTC)
    manifest = _build_manifest(encoded, specs, bench_started_at, now, partial=partial)
    if skipped:
        manifest["skipped_specs"] = [asdict(s) for s in skipped]
    (out_path / "bench_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
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
    # Per-run table name: BenchSpec.table_name pins it; None => generate
    # a uuid-suffixed name so two concurrent run_bench() calls against
    # the same Postgres can't collide on `vdbbench_vectors`. Adapters
    # that don't expose `set_table_name` (qdrant/lancedb/chroma manage
    # their own collection isolation already) silently skip the call.
    table_name = spec.table_name or _generate_table_name()
    setter = getattr(spec.adapter, "set_table_name", None)
    if callable(setter):
        setter(table_name)
    try:
        spec.adapter.setup(encoded.dim, spec.params)
    except Exception:
        # setup() failed partway through — call cleanup_partial_setup() so
        # any on-disk state (lance directory, chroma directory) left behind
        # by the partial setup doesn't leak into the next run or pollute
        # teardown. teardown() is intentionally NOT called here because
        # service adapters (pgvector, qdrant) manage their cleanup inside
        # teardown() which assumes a completed setup; calling it after a
        # failed setup would be undefined behaviour for those adapters.
        cleanup = getattr(spec.adapter, "cleanup_partial_setup", None)
        if callable(cleanup):
            with contextlib.suppress(Exception):
                cleanup()
        raise
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
    *,
    partial: bool = False,
) -> dict[str, object]:
    """Build the `bench_manifest.json` payload.

    Captures everything a reviewer needs to verify what was actually
    run: schema version (so future readers can branch), encoded bundle
    fingerprint, encoder identity, per-adapter package versions
    (best-effort via importlib.metadata), and best-effort host metadata
    (no PII — just CPU/OS/python info that frames the latency numbers).

    ``partial=True`` is written when the run was interrupted mid-loop.
    It signals that ``summary.parquet`` and ``timings.parquet`` contain
    only the specs that completed before the failure; the rest were not
    run. Readers should inspect the row count rather than assuming all
    specs are present.
    """
    return {
        "schema_version": BENCH_MANIFEST_SCHEMA_VERSION,
        "partial": partial,
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
    # `lat["mean"]` may be exactly 0.0 for in-memory / exact adapters that
    # complete in sub-microsecond time; the previous truthiness guard
    # (`if lat["mean"]`) treated 0.0 as "no data" and surfaced
    # ``qps_estimate=0.0`` for adapters that should report effectively
    # infinite throughput. Now we discriminate "no data" from
    # "instantaneous": a missing mean (None / NaN / negative) maps to
    # ``qps=0.0`` (existing semantics), but a true 0.0 mean produces
    # ``inf`` so a downstream comparison plot doesn't anchor on a
    # spurious zero.
    if lat["mean"] is None or np.isnan(lat["mean"]) or lat["mean"] < 0:
        qps = 0.0
    elif lat["mean"] == 0.0:
        qps = float("inf")
    else:
        qps = 1000.0 / lat["mean"]
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
