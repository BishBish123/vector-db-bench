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
from vdbbench.adapters.base import IndexStats, IngestStats, VectorStoreAdapter
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
    """One row in the per-(db, params) summary table."""

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


@dataclass(frozen=True)
class BenchResult:
    """Per-query timings + per-(db, params) summary, both ready for parquet.

    `manifest` is a dict shaped like `bench_manifest.json` (see
    `_build_manifest()`); `save()` writes it next to the parquet files
    so a reviewer can verify which encoded bundle, encoder, and
    adapter versions produced the numbers.
    """

    timings: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object] = field(default_factory=dict)

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


def run_bench(
    encoded: EncodedBundle,
    specs: list[BenchSpec],
    *,
    progress: bool = False,
) -> BenchResult:
    """Run every spec against `encoded` and return per-query + summary tables.

    Each spec is independently driven through the full lifecycle:
        adapter.setup(dim, params)
        adapter.ingest(pids, passage_vectors)
        adapter.build_index()
        # warm-up queries (timings discarded)
        # measured queries x repeats
        adapter.teardown()

    A failure in any spec raises and aborts — the bench is meant to be
    re-runnable, not partially-resilient. Catch upstream if you want to
    keep going on partial failure.
    """
    pids: list[str] = encoded.bundle.passages["pid"].astype(str).tolist()
    qids: list[str] = encoded.bundle.queries["qid"].astype(str).tolist()
    qrel_index = build_qrel_index(encoded.bundle.qrels)

    timing_rows: list[QueryTiming] = []
    summary_rows: list[RunSummary] = []

    bench_started_at = _dt.datetime.now(_dt.UTC)
    for spec in specs:
        if progress:
            print(f"[bench] {spec.display_label()} — setup", flush=True)
        spec.adapter.setup(encoded.dim, spec.params)
        try:
            ingest_stats = spec.adapter.ingest(pids, encoded.passage_vectors)
            index_stats = spec.adapter.build_index()

            # Warm-up — exercise the cache + kick the JIT path before timing.
            warmup_n = min(spec.warmup_queries, len(qids))
            for i in range(warmup_n):
                spec.adapter.search(encoded.query_vectors[i], spec.k)

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

            summary_rows.append(
                _build_summary(
                    spec, encoded, ingest_stats, index_stats, recalls, ndcgs, latencies_ms
                )
            )
        finally:
            spec.adapter.teardown()
    bench_completed_at = _dt.datetime.now(_dt.UTC)

    manifest = _build_manifest(encoded, specs, bench_started_at, bench_completed_at)
    return BenchResult(
        timings=pd.DataFrame([_qt_to_row(t) for t in timing_rows]),
        summary=pd.DataFrame([asdict(r) for r in summary_rows]),
        manifest=manifest,
    )


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


def _build_summary(
    spec: BenchSpec,
    encoded: EncodedBundle,
    ingest: IngestStats,
    index: IndexStats,
    recalls: list[float],
    ndcgs: list[float],
    latencies_ms: list[float],
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
    )
