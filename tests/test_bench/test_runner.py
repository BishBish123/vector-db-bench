"""Bench runner tests using a fake in-memory adapter."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vdbbench.adapters.base import IndexStats, IngestStats
from vdbbench.bench.runner import BenchSpec, run_bench
from vdbbench.corpus.bundle import CorpusBundle
from vdbbench.embed.encoder import EncodedBundle, FakeEncoder, encode_corpus

# ---------------------------------------------------------------------------
# A trivial in-memory exact-NN adapter for harness tests.
# ---------------------------------------------------------------------------


class _MemAdapter:
    name = "mem"

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
        if self._dim is None:
            raise RuntimeError("ingest before setup")
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
# Fixtures
# ---------------------------------------------------------------------------


def _toy_encoded() -> EncodedBundle:
    """A tiny bundle whose qrels match the brute-force top-1 of the encoder."""
    bundle = CorpusBundle(
        name="toy",
        passages=pd.DataFrame(
            {"pid": [f"p{i}" for i in range(8)], "text": [f"doc {i}" for i in range(8)]}
        ),
        queries=pd.DataFrame({"qid": ["q0", "q1"], "text": ["doc 0", "doc 3"]}),
        # The fake encoder is text-deterministic, so each query's gold pid is
        # the passage with the same text.
        qrels=pd.DataFrame({"qid": ["q0", "q1"], "pid": ["p0", "p3"], "relevance": [1, 1]}),
    )
    return encode_corpus(bundle, FakeEncoder(dim=16))


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestBenchSpec:
    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            BenchSpec(adapter=_MemAdapter(), k=0)

    def test_invalid_repeats(self) -> None:
        with pytest.raises(ValueError, match="repeats"):
            BenchSpec(adapter=_MemAdapter(), repeats=0)

    def test_negative_warmup(self) -> None:
        # `-1` is the sentinel for "use the profile default" — pick `-2` to
        # actually trip the validation.
        with pytest.raises(ValueError, match="warmup"):
            BenchSpec(adapter=_MemAdapter(), warmup_queries=-2)

    def test_params_hash_stable(self) -> None:
        a = BenchSpec(adapter=_MemAdapter(), params={"a": 1, "b": 2})
        b = BenchSpec(adapter=_MemAdapter(), params={"b": 2, "a": 1})
        assert a.params_hash() == b.params_hash()


# ---------------------------------------------------------------------------
# run_bench end-to-end
# ---------------------------------------------------------------------------


class TestRunBench:
    def test_one_spec_round_trip(self) -> None:
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=3, warmup_queries=1, repeats=1)
        result = run_bench(encoded, [spec])

        # 2 queries x 1 repeat = 2 timing rows.
        assert len(result.timings) == 2
        assert set(result.timings["db"]) == {"mem"}
        # Each row has k=3 retrieved pids.
        assert all(len(row) == 3 for row in result.timings["retrieved_pids"])
        # Summary has one row per (db, params).
        assert len(result.summary) == 1
        row = result.summary.iloc[0]
        assert row["db"] == "mem"
        assert row["n_queries"] == 2
        assert row["n_passages"] == 8
        assert row["dim"] == 16
        # Recall should be 1.0 — _MemAdapter is exact, the qrels match the
        # encoder's deterministic top-1 hit.
        assert row["recall_at_k_mean"] == pytest.approx(1.0)
        assert row["latency_ms_mean"] >= 0

    def test_multiple_specs_each_get_summary_row(self) -> None:
        encoded = _toy_encoded()
        specs = [
            BenchSpec(adapter=_MemAdapter(), params={"variant": "a"}, k=2),
            BenchSpec(adapter=_MemAdapter(), params={"variant": "b"}, k=2),
        ]
        result = run_bench(encoded, specs)
        assert len(result.summary) == 2
        assert set(result.summary["params_hash"]) == {
            specs[0].params_hash(),
            specs[1].params_hash(),
        }

    def test_repeats_multiplied_into_timing_count(self) -> None:
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=2, warmup_queries=0, repeats=3)
        result = run_bench(encoded, [spec])
        assert len(result.timings) == 6  # 2 queries x 3 repeats

    def test_failure_in_one_spec_aborts(self) -> None:
        encoded = _toy_encoded()

        class BadAdapter(_MemAdapter):
            name = "bad"

            def ingest(
                self, ids: list[str], vectors: np.ndarray, batch_size: int = 1024
            ) -> IngestStats:
                raise RuntimeError("simulated failure")

        spec = BenchSpec(adapter=BadAdapter(), k=2)
        with pytest.raises(RuntimeError, match="simulated failure"):
            run_bench(encoded, [spec])

    def test_tolerate_failures_skips_unreachable_adapter(self) -> None:
        """`--all` mode wraps each spec; a connection error on one spec
        is logged as a structured skip and the run continues."""
        encoded = _toy_encoded()

        class UnreachableAdapter(_MemAdapter):
            name = "unreachable"

            def setup(self, dim: int, params: dict[str, object]) -> None:
                raise ConnectionRefusedError("connection refused: no service on :5433")

        good = BenchSpec(adapter=_MemAdapter(), params={"v": "good"}, k=2)
        bad = BenchSpec(adapter=UnreachableAdapter(), params={"v": "bad"}, k=2)
        result = run_bench(encoded, [bad, good], tolerate_failures=True)
        # Good adapter ran; bad adapter is in skipped.
        assert len(result.summary) == 1
        assert result.summary.iloc[0]["db"] == "mem"
        assert len(result.skipped) == 1
        assert result.skipped[0].db == "unreachable"
        assert result.skipped[0].error_type == "ConnectionRefusedError"
        assert "connection refused" in result.skipped[0].reason
        # Manifest carries the skipped spec list so the gap is auditable.
        assert "skipped_specs" in result.manifest

    def test_tolerate_failures_does_not_swallow_real_bugs(self) -> None:
        """A non-connection error is still a real bug — must propagate
        even in tolerant mode, otherwise we'd mask harness regressions."""
        encoded = _toy_encoded()

        class BuggyAdapter(_MemAdapter):
            name = "buggy"

            def build_index(self) -> IndexStats:
                raise RuntimeError("real bug, not a missing service")

        spec = BenchSpec(adapter=BuggyAdapter(), k=2)
        with pytest.raises(RuntimeError, match="real bug"):
            run_bench(encoded, [spec], tolerate_failures=True)

    def test_tolerate_failures_off_propagates_connection_error(self) -> None:
        """Default mode (single-adapter run) hard-fails on connection
        errors — there's no other adapter to keep going for."""
        encoded = _toy_encoded()

        class UnreachableAdapter(_MemAdapter):
            name = "unreachable"

            def setup(self, dim: int, params: dict[str, object]) -> None:
                raise ConnectionRefusedError("nope")

        spec = BenchSpec(adapter=UnreachableAdapter(), k=2)
        with pytest.raises(ConnectionRefusedError):
            run_bench(encoded, [spec])

    def test_save_writes_parquet(self, tmp_path: Path) -> None:
        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        out = result.save(tmp_path / "bench-out")
        assert (out / "timings.parquet").exists()
        assert (out / "summary.parquet").exists()
        # Round-trip the summary so the schema is at least valid parquet.
        round_tripped = pd.read_parquet(out / "summary.parquet")
        assert list(round_tripped.columns) == list(result.summary.columns)

    def test_bench_writes_manifest_alongside_summary(self, tmp_path: Path) -> None:
        """Every bench run drops a `bench_manifest.json` next to the parquet
        files so a reviewer can verify which encoded bundle, encoder, and
        adapter versions produced the numbers."""
        import json as _json  # noqa: PLC0415

        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        out = result.save(tmp_path / "bench-out")
        manifest_path = out / "bench_manifest.json"
        assert manifest_path.exists()
        manifest = _json.loads(manifest_path.read_text())
        # Required fields present and well-typed.
        for key in (
            "schema_version",
            "encoded_bundle_fingerprint",
            "encoder_name",
            "encoder_dim",
            "adapter_versions",
            "host_metadata",
            "bench_started_at",
            "bench_completed_at",
            "bench_specs",
        ):
            assert key in manifest, f"missing manifest key {key!r}"

    def test_manifest_records_encoded_bundle_fingerprint(self) -> None:
        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        assert result.manifest["encoded_bundle_fingerprint"] == encoded.bundle.fingerprint()
        assert result.manifest["encoder_name"] == encoded.encoder_name
        assert result.manifest["encoder_dim"] == encoded.dim

    def test_manifest_schema_version_starts_at_1(self) -> None:
        """The manifest schema version is the explicit contract for future
        readers — pin it so a bump is intentional."""
        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        assert result.manifest["schema_version"] == 1

    def test_run_bench_records_memory_columns(self) -> None:
        """Every memory column on RunSummary must be populated by the runner.

        Adapters expose memory_footprint_bytes() and the harness depends on
        psutil for RSS sampling — but the bench used to throw all of that
        on the floor. Pin the columns so a regression here fails loud.
        """
        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        for col in (
            "baseline_rss_bytes",
            "index_rss_bytes",
            "peak_rss_bytes",
            "adapter_memory_bytes",
        ):
            assert col in result.summary.columns, f"missing memory column {col!r}"
        # peak_rss should be at least baseline (process can only grow during
        # a synchronous bench run, modulo gc returning pages — assert >= 0
        # rather than > baseline so we don't flake on a freed-pages run).
        row = result.summary.iloc[0]
        assert int(row["baseline_rss_bytes"]) > 0
        assert int(row["peak_rss_bytes"]) >= 0

    def test_summary_parquet_has_memory_schema(self, tmp_path: Path) -> None:
        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        out = result.save(tmp_path / "bench-out")
        round_tripped = pd.read_parquet(out / "summary.parquet")
        for col in (
            "baseline_rss_bytes",
            "index_rss_bytes",
            "peak_rss_bytes",
            "adapter_memory_bytes",
        ):
            assert col in round_tripped.columns, f"summary parquet missing {col!r}"

    def test_latency_records_actual_time(self) -> None:
        """Latency must be measured per query, not zero."""
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(search_sleep_ms=2.0), k=2, warmup_queries=0)
        result = run_bench(encoded, [spec])
        # Each query slept ~2ms; allow some slack for scheduler jitter.
        assert (result.timings["latency_ms"] >= 1.5).all()

    def test_summary_includes_p99_and_profile(self) -> None:
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=2, warmup_queries=0, repeats=4)
        result = run_bench(encoded, [spec])
        row = result.summary.iloc[0]
        assert "latency_ms_p99" in result.summary.columns
        assert row["latency_ms_p99"] >= row["latency_ms_p95"]
        assert row["profile"] == "warm"


class TestBenchSpecProfile:
    def test_warm_is_default(self) -> None:
        spec = BenchSpec(adapter=_MemAdapter())
        assert spec.profile == "warm"
        assert spec.warmup_queries == 10
        assert spec.repeats == 1

    def test_cold_profile_zero_warmup(self) -> None:
        spec = BenchSpec(adapter=_MemAdapter(), profile="cold")
        assert spec.warmup_queries == 0
        assert spec.repeats == 1

    def test_p99_profile_high_warmup_and_repeats(self) -> None:
        spec = BenchSpec(adapter=_MemAdapter(), profile="p99")
        assert spec.warmup_queries == 50
        assert spec.repeats == 5

    def test_explicit_warmup_overrides_profile(self) -> None:
        spec = BenchSpec(adapter=_MemAdapter(), profile="p99", warmup_queries=2)
        assert spec.warmup_queries == 2
        # repeats still pulls from the profile default since not overridden.
        assert spec.repeats == 5

    def test_unknown_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown profile"):
            BenchSpec(adapter=_MemAdapter(), profile="bogus")

    def test_profile_recorded_in_summary(self) -> None:
        encoded = _toy_encoded()
        spec = BenchSpec(adapter=_MemAdapter(), k=2, profile="cold")
        result = run_bench(encoded, [spec])
        assert result.summary.iloc[0]["profile"] == "cold"

    def test_profile_p99_yields_5_repeats_when_repeats_not_set(self) -> None:
        """The CLI passes the -1 sentinel for `repeats` so the profile picks
        the default. With profile=p99 that's 5 repeats — used to be silently
        clamped to 1 because the CLI hard-coded `repeats=1`.
        """
        spec = BenchSpec(adapter=_MemAdapter(), profile="p99", repeats=-1)
        assert spec.repeats == 5

    def test_explicit_repeats_overrides_profile(self) -> None:
        """Explicit `repeats=N` (any positive int) wins over the profile default."""
        spec = BenchSpec(adapter=_MemAdapter(), profile="p99", repeats=2)
        assert spec.repeats == 2
        spec_warm = BenchSpec(adapter=_MemAdapter(), profile="warm", repeats=7)
        assert spec_warm.repeats == 7
