"""Tests for the encoder protocol, FakeEncoder, and the encoding pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vdbbench.corpus.bundle import CorpusBundle, IncompatibleManifestError
from vdbbench.embed.encoder import (
    ENCODED_MANIFEST_SCHEMA_VERSION,
    EncodedBundle,
    Encoder,
    FakeEncoder,
    detect_device,
    encode_corpus,
    load_encoded_bundle,
)


def _toy_bundle() -> CorpusBundle:
    return CorpusBundle(
        name="toy",
        passages=pd.DataFrame({"pid": ["p0", "p1", "p2"], "text": ["alpha", "bravo", "charlie"]}),
        queries=pd.DataFrame({"qid": ["q0", "q1"], "text": ["alpha?", "bravo?"]}),
        qrels=pd.DataFrame({"qid": ["q0", "q1"], "pid": ["p0", "p1"], "relevance": [1, 1]}),
    )


# ---------------------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------------------


class TestDetectDevice:
    def test_returns_one_of_known_devices(self) -> None:
        assert detect_device() in {"cpu", "mps", "cuda"}


# ---------------------------------------------------------------------------
# FakeEncoder
# ---------------------------------------------------------------------------


class TestFakeEncoder:
    def test_satisfies_protocol(self) -> None:
        enc: Encoder = FakeEncoder()
        assert enc.name
        assert enc.dim > 0

    def test_shape_and_dtype(self) -> None:
        enc = FakeEncoder(dim=16)
        out = enc.encode(["a", "b", "c"])
        assert out.shape == (3, 16)
        assert out.dtype == np.float32

    def test_unit_norm(self) -> None:
        enc = FakeEncoder(dim=8)
        out = enc.encode(["a", "b", "c"])
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

    def test_deterministic(self) -> None:
        enc = FakeEncoder()
        a = enc.encode(["alpha", "bravo"])
        b = enc.encode(["alpha", "bravo"])
        np.testing.assert_array_equal(a, b)

    def test_text_changes_vector(self) -> None:
        enc = FakeEncoder()
        a = enc.encode(["alpha"])
        b = enc.encode(["bravo"])
        assert not np.allclose(a[0], b[0])

    def test_empty_input(self) -> None:
        enc = FakeEncoder(dim=4)
        out = enc.encode([])
        assert out.shape == (0, 4)
        assert out.dtype == np.float32

    def test_batch_size_invariant(self) -> None:
        """Batched and unbatched encoding must produce identical output."""
        enc = FakeEncoder()
        texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
        small_batch = enc.encode(texts, batch_size=2)
        big_batch = enc.encode(texts, batch_size=100)
        np.testing.assert_array_equal(small_batch, big_batch)

    def test_invalid_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            FakeEncoder(dim=0)
        with pytest.raises(ValueError, match="dim"):
            FakeEncoder(dim=128)  # over the cap

    def test_invalid_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            FakeEncoder().encode(["a"], batch_size=0)


# ---------------------------------------------------------------------------
# encode_corpus
# ---------------------------------------------------------------------------


class TestEncodeCorpus:
    def test_shapes_match_bundle(self) -> None:
        bundle = _toy_bundle()
        enc = FakeEncoder(dim=16)
        result = encode_corpus(bundle, enc)
        assert result.passage_vectors.shape == (bundle.n_passages, 16)
        assert result.query_vectors.shape == (bundle.n_queries, 16)
        assert result.dim == 16
        assert result.encoder_name == enc.name

    def test_metadata_includes_dim_and_batch(self) -> None:
        bundle = _toy_bundle()
        enc = FakeEncoder(dim=16)
        result = encode_corpus(bundle, enc, batch_size=8, metadata={"foo": "bar"})
        assert result.metadata["encoder_dim"] == 16
        assert result.metadata["batch_size"] == 8
        assert result.metadata["foo"] == "bar"

    def test_row_alignment_preserved(self) -> None:
        """Vector i must correspond to text i — fundamental invariant."""
        bundle = _toy_bundle()
        enc = FakeEncoder(dim=16)
        result = encode_corpus(bundle, enc)
        for i, text in enumerate(bundle.passages["text"]):
            np.testing.assert_array_equal(result.passage_vectors[i], enc.encode([text])[0])

    def test_invalid_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            encode_corpus(_toy_bundle(), FakeEncoder(), batch_size=0)

    def test_misbehaving_encoder_caught(self) -> None:
        """If an encoder lies about row count we surface it loudly."""

        class BadEncoder:
            name = "bad"
            dim = 4

            def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
                return np.zeros((len(texts) - 1, self.dim), dtype=np.float32)

        with pytest.raises(RuntimeError, match="row alignment"):
            encode_corpus(_toy_bundle(), BadEncoder())

    def test_encode_corpus_rejects_wrong_dim_passages(self) -> None:
        """Encoder advertises dim=N but returns shape (..., M) — fail loud."""

        class LyingEncoder:
            name = "liar"
            dim = 16  # advertised

            def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
                # Actually return half the advertised width.
                return np.zeros((len(texts), 8), dtype=np.float32)

        with pytest.raises(ValueError, match=r"disagrees with encoder\.dim=16"):
            encode_corpus(_toy_bundle(), LyingEncoder())

    def test_encode_corpus_rejects_nan_in_output(self) -> None:
        """An encoder producing NaN/Inf must be caught at encode time, not
        propagated into adapters where the failure mode is subtler."""

        class NanEncoder:
            name = "nan"
            dim = 4

            def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
                out = np.zeros((len(texts), self.dim), dtype=np.float32)
                out[0, 0] = float("nan")
                return out

        with pytest.raises(ValueError, match="non-finite"):
            encode_corpus(_toy_bundle(), NanEncoder())


# ---------------------------------------------------------------------------
# EncodedBundle invariants and round-trip
# ---------------------------------------------------------------------------


class TestEncodedBundleInvariants:
    def test_rejects_wrong_passage_count(self) -> None:
        bundle = _toy_bundle()
        with pytest.raises(ValueError, match="passage_vectors row count"):
            EncodedBundle(
                bundle=bundle,
                passage_vectors=np.zeros((bundle.n_passages + 1, 4), dtype=np.float32),
                query_vectors=np.zeros((bundle.n_queries, 4), dtype=np.float32),
                encoder_name="x",
            )

    def test_rejects_wrong_query_count(self) -> None:
        bundle = _toy_bundle()
        with pytest.raises(ValueError, match="query_vectors row count"):
            EncodedBundle(
                bundle=bundle,
                passage_vectors=np.zeros((bundle.n_passages, 4), dtype=np.float32),
                query_vectors=np.zeros((bundle.n_queries + 1, 4), dtype=np.float32),
                encoder_name="x",
            )

    def test_rejects_dim_mismatch(self) -> None:
        bundle = _toy_bundle()
        with pytest.raises(ValueError, match="vector dims disagree"):
            EncodedBundle(
                bundle=bundle,
                passage_vectors=np.zeros((bundle.n_passages, 4), dtype=np.float32),
                query_vectors=np.zeros((bundle.n_queries, 8), dtype=np.float32),
                encoder_name="x",
            )

    def test_rejects_non_2d_arrays(self) -> None:
        bundle = _toy_bundle()
        with pytest.raises(ValueError, match="passage_vectors must be 2-D"):
            EncodedBundle(
                bundle=bundle,
                passage_vectors=np.zeros(bundle.n_passages, dtype=np.float32),
                query_vectors=np.zeros((bundle.n_queries, 4), dtype=np.float32),
                encoder_name="x",
            )

    def test_widens_dtype_to_float32(self) -> None:
        bundle = _toy_bundle()
        encoded = EncodedBundle(
            bundle=bundle,
            passage_vectors=np.zeros((bundle.n_passages, 4), dtype=np.float64),
            query_vectors=np.zeros((bundle.n_queries, 4), dtype=np.float64),
            encoder_name="x",
        )
        assert encoded.passage_vectors.dtype == np.float32
        assert encoded.query_vectors.dtype == np.float32


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=16))
        out = encoded.save(tmp_path / "encoded")
        loaded = load_encoded_bundle(out)

        np.testing.assert_array_equal(loaded.passage_vectors, encoded.passage_vectors)
        np.testing.assert_array_equal(loaded.query_vectors, encoded.query_vectors)
        assert loaded.encoder_name == encoded.encoder_name
        assert loaded.dim == encoded.dim
        assert loaded.bundle.fingerprint() == bundle.fingerprint()

    def test_load_detects_id_drift(self, tmp_path: Path) -> None:
        """Mutating the parquet vector ids out from under the bundle must fail."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")

        # Tamper with passages.parquet to drop the first id.
        import pyarrow.parquet as pq  # noqa: PLC0415

        table = pq.read_table(out / "passages.parquet")
        tampered = table.slice(1)
        pq.write_table(tampered, out / "passages.parquet")

        with pytest.raises(ValueError, match="ids do not match"):
            load_encoded_bundle(out)

    def test_load_detects_swapped_corpus(self, tmp_path: Path) -> None:
        """Replacing `corpus/` with a different bundle (same ids) must fail."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")

        # Build a different bundle that re-uses the same ids but mutates text.
        swapped = CorpusBundle(
            name=bundle.name,
            passages=pd.DataFrame(
                {"pid": ["p0", "p1", "p2"], "text": ["MUTATED", "ALSO", "CHANGED"]}
            ),
            queries=bundle.queries,
            qrels=bundle.qrels,
        )
        # Overwrite the corpus saved alongside the vectors.
        swapped.save(out / "corpus")

        with pytest.raises(ValueError, match="corpus fingerprint does not match"):
            load_encoded_bundle(out)

    def test_save_includes_schema_version(self, tmp_path: Path) -> None:
        """Every fresh encoded-bundle save stamps the current schema_version."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema_version"] == ENCODED_MANIFEST_SCHEMA_VERSION

    def test_load_uses_memory_map(self, tmp_path: Path) -> None:
        """Vector reads pass `memory_map=True` to pyarrow so the parquet's
        data pages are mmap'd instead of copied into a fresh buffer.
        Disk-resident case is the common one; the network-FS slowdown
        is documented in the loader comment.
        """
        from unittest.mock import patch  # noqa: PLC0415

        import pyarrow.parquet as pq  # noqa: PLC0415

        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")

        original = pq.read_table
        captured: list[tuple[Path, dict[str, object]]] = []

        def spy(path: object, *args: object, **kwargs: object) -> object:
            captured.append((Path(str(path)), dict(kwargs)))
            return original(path, *args, **kwargs)

        with patch("vdbbench.embed.encoder.pq.read_table", spy):
            load_encoded_bundle(out)

        # Only the *vector* parquet reads (passages.parquet / queries.parquet
        # at the encoded-bundle root) are expected to opt into memory_map;
        # corpus reads go through pandas and aren't load-bearing here.
        vector_calls = [
            kwargs
            for path, kwargs in captured
            if path.name in {"passages.parquet", "queries.parquet"}
            and path.parent == out
        ]
        assert vector_calls, "expected pq.read_table to be called for vector files"
        for call_kwargs in vector_calls:
            assert call_kwargs.get("memory_map") is True, (
                f"vector pq.read_table was called without memory_map=True: {call_kwargs}"
            )

    def test_load_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        """An encoded-bundle manifest with a future/unknown schema_version
        must raise — silently mis-parsing would attach the wrong vector
        format to a corpus."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = 99
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(IncompatibleManifestError, match="schema_version=99"):
            load_encoded_bundle(out)

    def test_load_warns_on_encoded_at_fingerprint_drift(self, tmp_path: Path) -> None:
        """If the manifest's encoded-at fingerprint disagrees with the loaded
        corpus we warn — distinct from the hard `bundle_fingerprint` check
        because the manifest may have been hand-edited or partially-stale.
        """
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "encoded")

        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        # Drift only the soft check — leave `bundle_fingerprint` consistent
        # so the hard check passes and we exercise the warning path.
        manifest["encoded_at_fingerprint"] = "deadbeef" * 4
        manifest_path.write_text(json.dumps(manifest))

        with pytest.warns(UserWarning, match="corpus has changed since encoding"):
            load_encoded_bundle(out)

    def test_load_uses_zero_copy_path(self, tmp_path: Path) -> None:
        """Reading vectors back must avoid the `to_pylist()` Python detour.

        We can't directly assert "no Python list was allocated", but we can
        verify the loaded matrix has the right dtype, shape, and contiguity
        — the three things the new buffer path is supposed to guarantee
        — and that a moderately large round-trip is fast (<500ms locally
        for 5k x 384 = ~7.5 MB; the slow `to_pylist` path used to take
        seconds for the same size).
        """
        import time as _time  # noqa: PLC0415

        n, dim = 5_000, 384
        bundle = CorpusBundle(
            name="zero-copy",
            passages=pd.DataFrame(
                {
                    "pid": [f"p{i:05d}" for i in range(n)],
                    "text": [f"d{i}" for i in range(n)],
                }
            ),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p00000"], "relevance": [1]}),
        )
        # FakeEncoder caps dim at 64; build vectors directly so the test
        # exercises a realistic embedding width.
        rng = np.random.default_rng(0)
        passage_vecs = rng.standard_normal((n, dim)).astype(np.float32)
        query_vecs = rng.standard_normal((1, dim)).astype(np.float32)
        encoded = EncodedBundle(
            bundle=bundle,
            passage_vectors=passage_vecs,
            query_vectors=query_vecs,
            encoder_name="rand",
        )
        out = encoded.save(tmp_path / "zerocopy")

        t0 = _time.perf_counter()
        loaded = load_encoded_bundle(out)
        elapsed = _time.perf_counter() - t0

        assert loaded.passage_vectors.shape == (n, dim)
        assert loaded.passage_vectors.dtype == np.float32
        assert loaded.passage_vectors.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(loaded.passage_vectors, encoded.passage_vectors)
        # Generous bound — the buffer path should finish in well under
        # 500ms on a laptop. The pylist path used to take >2s here.
        assert elapsed < 2.0, f"load took {elapsed:.2f}s — pylist regression?"

    def test_save_does_not_materialize_full_matrix_as_python(self, tmp_path: Path) -> None:
        """Regression: `vecs.tolist()` would OOM on benchmark-sized matrices.

        We can't easily count Python objects, but we can verify the buffer
        path works for a non-trivial matrix and the round-trip is exact.
        """
        bundle = CorpusBundle(
            name="big",
            passages=pd.DataFrame(
                {
                    "pid": [f"p{i:04d}" for i in range(2_000)],
                    "text": [f"d{i}" for i in range(2_000)],
                }
            ),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0000"], "relevance": [1]}),
        )
        encoded = encode_corpus(bundle, FakeEncoder(dim=64))
        out = encoded.save(tmp_path / "big")
        loaded = load_encoded_bundle(out)
        np.testing.assert_array_equal(loaded.passage_vectors, encoded.passage_vectors)


class TestEncodedBundleEdgeCases:
    def test_empty_query_side_round_trips(self, tmp_path: Path) -> None:
        """Bundle with zero queries must save and load (regression for `np.array([])`)."""
        bundle = CorpusBundle(
            name="no-queries",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": [], "text": []}, dtype=object),
            qrels=pd.DataFrame(
                {
                    "qid": pd.Series([], dtype=str),
                    "pid": pd.Series([], dtype=str),
                    "relevance": pd.Series([], dtype=float),
                }
            ),
        )
        encoded = EncodedBundle(
            bundle=bundle,
            passage_vectors=np.zeros((1, 8), dtype=np.float32),
            query_vectors=np.zeros((0, 8), dtype=np.float32),
            encoder_name="fake",
        )
        out = encoded.save(tmp_path / "empty-queries")
        loaded = load_encoded_bundle(out)
        assert loaded.query_vectors.shape == (0, 8)

    def test_path_metadata_is_serialisable(self, tmp_path: Path) -> None:
        """Common provenance values like Path / np.int64 must not break save."""
        bundle = _toy_bundle()
        encoded = encode_corpus(
            bundle,
            FakeEncoder(dim=8),
            metadata={"src": Path("/tmp/foo"), "n": np.int64(5)},
        )
        out = encoded.save(tmp_path / "with-metadata")
        loaded = load_encoded_bundle(out)
        assert loaded.metadata["src"] == "/tmp/foo"
        assert loaded.metadata["n"] == 5

    def test_save_rejects_post_encoding_mutation(self, tmp_path: Path) -> None:
        """If callers mutate the bundle after encoding, save must fail loudly."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        bundle.passages.loc[0, "text"] = "MUTATED AFTER ENCODE"
        with pytest.raises(RuntimeError, match="mutated in place"):
            encoded.save(tmp_path / "drift")

    def test_load_rejects_dim_mismatch_in_vector_files(self, tmp_path: Path) -> None:
        """Swapping in a different encoder's vectors (different dim) must fail."""
        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "ok")

        # Re-encode the same bundle with a different dim and overwrite.
        encoded2 = encode_corpus(bundle, FakeEncoder(dim=16))
        # Persist the wider vectors directly into the existing tree.
        from vdbbench.embed.encoder import _write_vectors  # noqa: PLC0415

        _write_vectors(encoded2.passage_vectors, bundle.passages["pid"], out / "passages.parquet")
        with pytest.raises(ValueError, match="dim 16 but the manifest declares dim 8"):
            load_encoded_bundle(out)

    def test_manifest_records_encoded_at_fingerprint(self, tmp_path: Path) -> None:
        """The on-disk manifest must record the bundle fingerprint at encode time."""
        import json  # noqa: PLC0415

        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=8))
        out = encoded.save(tmp_path / "ok")
        manifest = json.loads((out / "manifest.json").read_text())
        assert "encoded_at_fingerprint" in manifest
        assert manifest["encoded_at_fingerprint"] == bundle.fingerprint()
        # Sanity: the dedicated drift-detection field equals the corpus's
        # bundle fingerprint when no mutation occurred.
        assert manifest["encoded_at_fingerprint"] == manifest["bundle_fingerprint"]

    def test_manifest_records_encoder_dim_and_name(self, tmp_path: Path) -> None:
        import json  # noqa: PLC0415

        bundle = _toy_bundle()
        encoded = encode_corpus(bundle, FakeEncoder(dim=12))
        out = encoded.save(tmp_path / "names")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["dim"] == 12
        assert manifest["encoder_name"] == "fake-test-encoder"
