"""Unit tests for retrieval metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vdbbench.metrics.retrieval import (
    RetrievalResult,
    aggregate,
    build_qrel_index,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_basic(self) -> None:
        r = RetrievalResult(qid="q1", retrieved_pids=("p0", "p1", "p2"))
        assert r.qid == "q1"
        assert r.retrieved_pids == ("p0", "p1", "p2")

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            RetrievalResult(qid="q1", retrieved_pids=("p0", "p1", "p0"))

    def test_empty_retrieved_is_allowed(self) -> None:
        # A DB returning no results is valid (recall@k = 0 for that query).
        RetrievalResult(qid="q1", retrieved_pids=())


# ---------------------------------------------------------------------------
# build_qrel_index
# ---------------------------------------------------------------------------


class TestBuildQrelIndex:
    def test_groups_by_qid(self) -> None:
        df = pd.DataFrame(
            {
                "qid": ["q1", "q1", "q2"],
                "pid": ["p0", "p1", "p2"],
                "relevance": [1.0, 2.0, 1.0],
            }
        )
        idx = build_qrel_index(df)
        assert idx == {"q1": {"p0": 1.0, "p1": 2.0}, "q2": {"p2": 1.0}}

    def test_drops_zero_relevance(self) -> None:
        """relevance==0 means 'judged not relevant' — must not count as positive."""
        df = pd.DataFrame(
            {
                "qid": ["q1", "q1"],
                "pid": ["p0", "p1"],
                "relevance": [1.0, 0.0],
            }
        )
        idx = build_qrel_index(df)
        assert idx == {"q1": {"p0": 1.0}}

    def test_empty_input(self) -> None:
        df = pd.DataFrame(columns=["qid", "pid", "relevance"])
        assert build_qrel_index(df) == {}

    def test_int_columns_coerced(self) -> None:
        df = pd.DataFrame({"qid": [1], "pid": [10], "relevance": [1]})
        idx = build_qrel_index(df)
        assert idx == {"1": {"10": 1.0}}


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ("p0", "p1", "p2", "p3")
        relevant = {"p0": 1.0, "p1": 1.0}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ("p0", "x", "p1")
        relevant = {"p0": 1.0, "p1": 1.0}
        # k=2 catches p0 only — recall = 1/2
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_zero_recall(self) -> None:
        retrieved = ("x", "y", "z")
        relevant = {"p0": 1.0}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ("p0",)
        relevant = {"p0": 1.0, "p1": 1.0}
        # Only 1 of 2 positives shown — recall = 0.5
        assert recall_at_k(retrieved, relevant, k=10) == 0.5

    def test_no_positives_returns_zero(self) -> None:
        assert recall_at_k(("p0",), {}, k=5) == 0.0

    def test_invalid_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            recall_at_k(("p0",), {"p0": 1.0}, k=0)

    def test_list_input_works(self) -> None:
        # Accept list as well as tuple.
        assert recall_at_k(["p0"], {"p0": 1.0}, k=1) == 1.0

    def test_zero_grade_judged_negatives_are_ignored(self) -> None:
        """Regression: pids with relevance == 0 must not count as positives."""
        # Only p1 is relevant. Retrieving the judged-negative p0 should miss.
        assert recall_at_k(("p0",), {"p0": 0.0, "p1": 1.0}, k=1) == 0.0
        # And retrieving the actual positive should recall 1/1.
        assert recall_at_k(("p1",), {"p0": 0.0, "p1": 1.0}, k=1) == 1.0


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit(self) -> None:
        assert hit_rate(("p0", "p1"), {"p1": 1.0}, k=2) == 1.0

    def test_miss(self) -> None:
        assert hit_rate(("p0", "p1"), {"p2": 1.0}, k=2) == 0.0

    def test_only_counts_top_k(self) -> None:
        # Hit is at position 3, k=2 misses it.
        assert hit_rate(("a", "b", "p0"), {"p0": 1.0}, k=2) == 0.0
        assert hit_rate(("a", "b", "p0"), {"p0": 1.0}, k=3) == 1.0

    def test_zero_grade_does_not_count_as_hit(self) -> None:
        """Regression: judged-negative pids are not hits."""
        assert hit_rate(("p0",), {"p0": 0.0, "p1": 1.0}, k=1) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr(("p0", "x"), {"p0": 1.0}) == 1.0

    def test_third_position(self) -> None:
        assert mrr(("a", "b", "p0"), {"p0": 1.0}) == 1.0 / 3

    def test_no_hit(self) -> None:
        assert mrr(("a", "b"), {"p0": 1.0}) == 0.0

    def test_capped_by_k(self) -> None:
        # Hit at rank 5; with k=3 we see no hit.
        retrieved = ("a", "b", "c", "d", "p0")
        assert mrr(retrieved, {"p0": 1.0}, k=3) == 0.0
        assert mrr(retrieved, {"p0": 1.0}, k=5) == 1.0 / 5

    def test_zero_grade_skipped(self) -> None:
        """Regression: judged-negative pids do not stop the search."""
        # p0 is judged-negative so the search should continue and find p1.
        assert mrr(("p0", "p1"), {"p0": 0.0, "p1": 1.0}) == 0.5


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        retrieved = ("p_high", "p_low")
        relevant = {"p_high": 3.0, "p_low": 1.0}
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_inverted_ranking_below_perfect(self) -> None:
        retrieved = ("p_low", "p_high")
        relevant = {"p_high": 3.0, "p_low": 1.0}
        score = ndcg_at_k(retrieved, relevant, k=2)
        assert 0.0 < score < 1.0

    def test_no_positives_returns_zero(self) -> None:
        assert ndcg_at_k(("a",), {}, k=5) == 0.0

    def test_no_relevant_in_top_k(self) -> None:
        assert ndcg_at_k(("a", "b"), {"p0": 1.0}, k=2) == 0.0

    def test_known_value(self) -> None:
        """A hand-checked NDCG value pins the formula."""
        retrieved = ("p0",)  # rank 1
        relevant = {"p0": 1.0}
        # DCG = (2^1 - 1) / log2(2) = 1.0
        # IDCG = 1.0
        # NDCG = 1.0
        assert ndcg_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)

    def test_graded_gain(self) -> None:
        """Graded relevance must show up in DCG (the 2^rel - 1 part)."""
        retrieved = ("p0",)
        relevant = {"p0": 2.0, "p1": 1.0}
        # DCG = (2^2 - 1) / log2(2) = 3
        # IDCG (top-1 of {2.0, 1.0} grades) = (2^2 - 1)/1 = 3
        # NDCG = 1.0
        assert ndcg_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
        # At k=2, IDCG includes the second positive too, but retrieved doesn't,
        # so NDCG drops.
        score = ndcg_at_k(retrieved, relevant, k=2)
        ideal = 3 + (2**1 - 1) / math.log2(3)
        assert score == pytest.approx(3 / ideal)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_known_distribution(self) -> None:
        # 5 values: [0.1, 0.5, 0.5, 0.5, 1.0]
        agg = aggregate([0.1, 0.5, 0.5, 0.5, 1.0])
        assert agg["n"] == 5
        assert agg["mean"] == pytest.approx(0.52)
        assert agg["p50"] == pytest.approx(0.5)
        # std with ddof=1
        assert agg["std"] == pytest.approx(np.std([0.1, 0.5, 0.5, 0.5, 1.0], ddof=1))

    def test_empty_input_nan(self) -> None:
        agg = aggregate([])
        assert agg["n"] == 0
        for key in ("mean", "p50", "p95", "std"):
            assert math.isnan(agg[key])

    def test_single_value_zero_std(self) -> None:
        agg = aggregate([0.7])
        assert agg["n"] == 1
        assert agg["std"] == 0.0
        assert agg["mean"] == pytest.approx(0.7)
