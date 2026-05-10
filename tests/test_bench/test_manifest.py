"""Tests for the bench_manifest.json read path.

The writer side lives in `BenchResult.save` (covered in test_runner.py).
This file pins the *reader* contract: `load_bench_manifest` must
round-trip a manifest written by `BenchResult.save`, refuse a file
without a recognisable `schema_version`, and refuse one stamped with a
mismatched version. Without these guards the schema_version field is
write-only — every release writes it, no release ever validates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdbbench.bench.runner import (
    BENCH_MANIFEST_SCHEMA_VERSION,
    IncompatibleBenchManifestError,
    load_bench_manifest,
)


def _write_manifest(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


class TestLoadBenchManifest:
    def test_round_trips_a_just_written_manifest(self, tmp_path: Path) -> None:
        """A manifest produced by BenchResult.save loads back identical."""
        # Run a tiny bench to produce a real manifest, then save + reload.
        # Imported here so the module-level imports stay tight.
        from tests.test_bench.test_runner import _MemAdapter, _toy_encoded  # noqa: PLC0415
        from vdbbench.bench.runner import BenchSpec, run_bench  # noqa: PLC0415

        encoded = _toy_encoded()
        result = run_bench(encoded, [BenchSpec(adapter=_MemAdapter(), k=2)])
        out = result.save(tmp_path / "bench-out")
        loaded = load_bench_manifest(out / "bench_manifest.json")

        assert loaded == result.manifest
        assert loaded["schema_version"] == BENCH_MANIFEST_SCHEMA_VERSION

    def test_load_bench_manifest_rejects_unknown_schema_version(
        self, tmp_path: Path
    ) -> None:
        """A future schema_version must fail closed, not be silently consumed.

        Today there are no backwards-compat shims; loading a v2 manifest
        with v1 code would let downstream analysis run on data the reader
        doesn't understand. The error names both versions so a caller can
        decide whether to upgrade vdbbench or pin to the producer's release.
        """
        future = BENCH_MANIFEST_SCHEMA_VERSION + 1
        path = _write_manifest(
            tmp_path / "manifest.json",
            {
                "schema_version": future,
                "encoded_bundle_fingerprint": "deadbeef",
                "encoder_name": "fake",
                "encoder_dim": 4,
                "adapter_versions": {},
                "host_metadata": {},
                "bench_specs": [],
            },
        )
        with pytest.raises(IncompatibleBenchManifestError) as excinfo:
            load_bench_manifest(path)
        assert excinfo.value.found == future
        assert excinfo.value.expected == BENCH_MANIFEST_SCHEMA_VERSION

    def test_load_bench_manifest_rejects_missing_schema_version(
        self, tmp_path: Path
    ) -> None:
        """A manifest with no schema_version at all is unauditable — it could
        have come from any vdbbench release. Refuse rather than guess."""
        path = _write_manifest(
            tmp_path / "manifest.json",
            {"encoded_bundle_fingerprint": "deadbeef", "encoder_name": "fake"},
        )
        with pytest.raises(IncompatibleBenchManifestError) as excinfo:
            load_bench_manifest(path)
        assert excinfo.value.found is None

    def test_load_bench_manifest_rejects_non_object_payload(
        self, tmp_path: Path
    ) -> None:
        """A JSON list / string / number isn't a manifest — fail loudly."""
        path = _write_manifest(tmp_path / "manifest.json", ["not", "a", "manifest"])
        with pytest.raises(IncompatibleBenchManifestError):
            load_bench_manifest(path)

    def test_bench_manifest_round_trip(self, tmp_path: Path) -> None:
        """Hand-written manifest with the current schema_version round-trips
        through load_bench_manifest unchanged. Pins the equality contract
        without depending on the bench runner to produce the bytes.
        """
        manifest = {
            "schema_version": BENCH_MANIFEST_SCHEMA_VERSION,
            "encoded_bundle_fingerprint": "deadbeef",
            "encoder_name": "synthetic-gaussian",
            "encoder_dim": 8,
            "adapter_versions": {"mem": "0.0.0"},
            "host_metadata": {"platform": "test"},
            "bench_started_at": "2026-05-09T00:00:00",
            "bench_completed_at": "2026-05-09T00:00:01",
            "vdbbench_version": "0.0.0",
            "bench_specs": [],
        }
        path = _write_manifest(tmp_path / "manifest.json", manifest)
        loaded = load_bench_manifest(path)
        assert loaded == manifest

    def test_load_bench_manifest_accepts_string_path(self, tmp_path: Path) -> None:
        """The signature advertises `str | Path`; both have to work."""
        path = _write_manifest(
            tmp_path / "manifest.json",
            {"schema_version": BENCH_MANIFEST_SCHEMA_VERSION},
        )
        loaded = load_bench_manifest(str(path))
        assert loaded == {"schema_version": BENCH_MANIFEST_SCHEMA_VERSION}
