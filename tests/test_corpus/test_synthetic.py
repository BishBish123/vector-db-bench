"""Unit tests for the synthetic corpus generator."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vdbbench.corpus.synthetic import SyntheticConfig, generate_synthetic


def _decode_vec(text: str, dim: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32).reshape(dim)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "msg"),
        [
            ({"n_passages": 0}, "n_passages"),
            ({"n_queries": 0}, "n_queries"),
            ({"dim": 0}, "dim"),
            ({"relevant_k": 0}, "relevant_k"),
            ({"n_passages": 5, "relevant_k": 6}, "relevant_k"),
        ],
    )
    def test_invalid_configs_rejected(self, kwargs: dict[str, int], msg: str) -> None:
        defaults = {"n_passages": 10, "n_queries": 5, "dim": 4, "relevant_k": 3, "seed": 0}
        defaults.update(kwargs)
        with pytest.raises(ValueError, match=msg):
            SyntheticConfig(**defaults)


class TestGenerate:
    def test_shape(self) -> None:
        cfg = SyntheticConfig(n_passages=20, n_queries=4, dim=8, relevant_k=3, seed=0)
        c = generate_synthetic(cfg)

        assert c.bundle.n_passages == 20
        assert c.bundle.n_queries == 4
        assert c.passage_vectors.shape == (20, 8)
        assert c.query_vectors.shape == (4, 8)
        # 4 queries x 3 relevant qrels each
        assert c.bundle.n_qrels == 12

    def test_vectors_are_unit_length(self) -> None:
        c = generate_synthetic(SyntheticConfig(n_passages=10, n_queries=3, dim=16, seed=1))
        np.testing.assert_allclose(np.linalg.norm(c.passage_vectors, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(c.query_vectors, axis=1), 1.0, atol=1e-6)

    def test_vectors_are_float32(self) -> None:
        """Qrels are derived from these vectors and the bundle text serializes
        them as float32 — the in-memory dtype must match or the round-trip
        breaks ground-truth correctness for near-tied neighbours."""
        c = generate_synthetic(SyntheticConfig(n_passages=10, n_queries=3, dim=16, seed=1))
        assert c.passage_vectors.dtype == np.float32
        assert c.query_vectors.dtype == np.float32

    def test_text_field_round_trips_to_vectors(self) -> None:
        cfg = SyntheticConfig(n_passages=8, n_queries=2, dim=4, relevant_k=2, seed=0)
        c = generate_synthetic(cfg)
        for i, text in enumerate(c.bundle.passages["text"]):
            np.testing.assert_array_equal(_decode_vec(text, cfg.dim), c.passage_vectors[i])
        for j, text in enumerate(c.bundle.queries["text"]):
            np.testing.assert_array_equal(_decode_vec(text, cfg.dim), c.query_vectors[j])

    def test_qrels_match_brute_force_topk(self) -> None:
        cfg = SyntheticConfig(n_passages=64, n_queries=8, dim=16, relevant_k=5, seed=7)
        c = generate_synthetic(cfg)

        # Recompute the brute-force top-relevant_k for every query and check it
        # matches the qrels (which encode the top-k as descending relevance).
        sims = c.query_vectors @ c.passage_vectors.T
        for q_i, qid in enumerate(c.bundle.queries["qid"]):
            ranked = np.argsort(-sims[q_i])[: cfg.relevant_k]
            expected_pids = [f"p{int(idx):08d}" for idx in ranked]
            row = c.bundle.qrels[c.bundle.qrels["qid"] == qid].sort_values(
                "relevance", ascending=False
            )
            assert list(row["pid"]) == expected_pids
            assert list(row["relevance"]) == list(range(cfg.relevant_k, 0, -1))

    def test_reproducible_with_same_seed(self) -> None:
        cfg = SyntheticConfig(n_passages=32, n_queries=4, dim=8, seed=42)
        a = generate_synthetic(cfg)
        b = generate_synthetic(cfg)
        np.testing.assert_array_equal(a.passage_vectors, b.passage_vectors)
        np.testing.assert_array_equal(a.query_vectors, b.query_vectors)
        assert a.bundle.fingerprint() == b.bundle.fingerprint()

    def test_different_seed_diverges(self) -> None:
        a = generate_synthetic(SyntheticConfig(n_passages=32, n_queries=4, dim=8, seed=1))
        b = generate_synthetic(SyntheticConfig(n_passages=32, n_queries=4, dim=8, seed=2))
        assert not np.array_equal(a.passage_vectors, b.passage_vectors)
        assert a.bundle.fingerprint() != b.bundle.fingerprint()

    def test_synthetic_corpus_equality_does_not_raise_on_ndarray(self) -> None:
        """Regression: dataclass-generated __eq__ on ndarray fields raises ValueError.

        We disable structural equality on SyntheticCorpus so that the comparison
        falls back to object identity instead of `np.ndarray.__eq__` (which
        returns an array and blows up when used as a bool).
        """
        cfg = SyntheticConfig(n_passages=10, n_queries=2, dim=4, relevant_k=2, seed=0)
        a = generate_synthetic(cfg)
        b = generate_synthetic(cfg)
        # The key invariant: comparison must not raise. Two distinct instances
        # may compare unequal (object identity) — that's fine.
        _ = a == b  # would raise ValueError under default dataclass __eq__
