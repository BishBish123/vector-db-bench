"""Tests for the BEIR adapter (pure layer — no network)."""

from __future__ import annotations

import pytest

from vdbbench.corpus.beir import (
    BeirQrel,
    BeirQuery,
    BeirRecord,
    _coerce_qrel,
    _coerce_query,
    _coerce_record,
    _stable_priority,
    _stream_sample_corpus,
    bundle_from_beir_records,
)

# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


class TestCoerceRecord:
    def test_beir_dict_with_id_and_text(self) -> None:
        rec = _coerce_record({"_id": "doc-7", "text": "hello"})
        assert rec == BeirRecord(pid="doc-7", text="hello")

    def test_concatenates_title_and_text(self) -> None:
        rec = _coerce_record({"_id": "x", "title": "Title", "text": "body"})
        assert rec.text == "Title. body"

    def test_strips_trailing_dot_when_body_empty(self) -> None:
        rec = _coerce_record({"_id": "x", "title": "Title", "text": ""})
        assert rec.text == "Title"

    def test_int_id_coerced_to_string(self) -> None:
        rec = _coerce_record({"_id": 42, "text": "x"})
        assert rec.pid == "42"

    def test_passthrough_for_existing_record(self) -> None:
        original = BeirRecord(pid="x", text="y")
        assert _coerce_record(original) is original

    @pytest.mark.parametrize("missing", [{"text": "x"}, {"_id": None, "text": "x"}])
    def test_missing_id_rejected(self, missing: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="missing id"):
            _coerce_record(missing)


class TestCoerceQuery:
    def test_basic(self) -> None:
        q = _coerce_query({"_id": "q1", "text": "what?"})
        assert q == BeirQuery(qid="q1", text="what?")

    def test_alternate_field_names(self) -> None:
        q = _coerce_query({"qid": "q1", "query": "what?"})
        assert q.qid == "q1"
        assert q.text == "what?"


class TestCoerceQrel:
    def test_beir_keys(self) -> None:
        qr = _coerce_qrel({"query-id": "q1", "corpus-id": "p1", "score": 1})
        assert qr == BeirQrel(qid="q1", pid="p1", relevance=1.0)

    def test_alternate_keys(self) -> None:
        qr = _coerce_qrel({"qid": "q1", "pid": "p1", "relevance": 0.5})
        assert qr.relevance == 0.5

    def test_negative_relevance_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative relevance"):
            _coerce_qrel({"query-id": "q", "corpus-id": "p", "score": -1})

    def test_missing_pid_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing qid or pid"):
            _coerce_qrel({"query-id": "q", "score": 1})


# ---------------------------------------------------------------------------
# bundle_from_beir_records
# ---------------------------------------------------------------------------


def _toy_corpus() -> tuple[list[dict], list[dict], list[dict]]:
    corpus = [
        {"_id": "p1", "title": "T1", "text": "doc one"},
        {"_id": "p2", "title": "", "text": "doc two"},
        {"_id": "p3", "title": "T3", "text": "doc three"},
    ]
    queries = [
        {"_id": "q1", "text": "what is one?"},
        {"_id": "q2", "text": "what is two?"},
    ]
    qrels = [
        {"query-id": "q1", "corpus-id": "p1", "score": 1},
        {"query-id": "q2", "corpus-id": "p2", "score": 1},
    ]
    return corpus, queries, qrels


class TestBundleFromBeir:
    def test_happy_path(self) -> None:
        corpus, queries, qrels = _toy_corpus()
        bundle = bundle_from_beir_records("toy", corpus, queries, qrels)
        assert bundle.n_passages == 3
        assert bundle.n_queries == 2
        assert bundle.n_qrels == 2
        # title concatenation worked
        assert bundle.passages.set_index("pid").loc["p1", "text"] == "T1. doc one"
        # source metadata is set
        assert bundle.metadata["source"] == "beir"

    def test_drops_orphan_qrels_by_default(self) -> None:
        corpus, queries, qrels = _toy_corpus()
        qrels.append({"query-id": "q1", "corpus-id": "GHOST", "score": 1})
        qrels.append({"query-id": "GHOST", "corpus-id": "p1", "score": 1})
        bundle = bundle_from_beir_records("toy", corpus, queries, qrels)
        # Orphan qrels are removed at the boundary so the bundle validators
        # don't reject the whole load.
        for _, row in bundle.qrels.iterrows():
            assert row["pid"] in set(bundle.passages["pid"])
            assert row["qid"] in set(bundle.queries["qid"])

    def test_orphans_propagated_when_drop_disabled(self) -> None:
        corpus, queries, qrels = _toy_corpus()
        qrels.append({"query-id": "q1", "corpus-id": "GHOST", "score": 1})
        with pytest.raises(ValueError, match="pids not in passages"):
            bundle_from_beir_records("toy", corpus, queries, qrels, drop_orphan_qrels=False)

    def test_dedupes_passages_and_queries(self) -> None:
        corpus, queries, qrels = _toy_corpus()
        corpus.append({"_id": "p1", "title": "DUP", "text": "should be ignored"})
        queries.append({"_id": "q1", "text": "DUP"})
        bundle = bundle_from_beir_records("toy", corpus, queries, qrels)
        # First-write-wins on duplicate ids — original text preserved.
        assert bundle.passages.set_index("pid").loc["p1", "text"] == "T1. doc one"
        assert bundle.queries.set_index("qid").loc["q1", "text"] == "what is one?"

    def test_dedupes_qrel_pairs(self) -> None:
        corpus, queries, qrels = _toy_corpus()
        qrels.append({"query-id": "q1", "corpus-id": "p1", "score": 2})  # duplicate
        bundle = bundle_from_beir_records("toy", corpus, queries, qrels)
        # First wins; bundle validators would reject duplicate (qid,pid) anyway.
        rel = bundle.qrels[(bundle.qrels["qid"] == "q1") & (bundle.qrels["pid"] == "p1")]
        assert len(rel) == 1
        assert rel["relevance"].iloc[0] == 1.0

    def test_supports_dataclass_inputs(self) -> None:
        bundle = bundle_from_beir_records(
            "toy",
            corpus=[BeirRecord("p1", "doc"), BeirRecord("p2", "doc2")],
            queries=[BeirQuery("q1", "?")],
            qrels=[BeirQrel("q1", "p1", 1.0)],
        )
        assert bundle.n_passages == 2

    def test_metadata_merges_with_source(self) -> None:
        bundle = bundle_from_beir_records(
            "toy",
            corpus=[BeirRecord("p1", "x")],
            queries=[BeirQuery("q1", "?")],
            qrels=[BeirQrel("q1", "p1", 1.0)],
            metadata={"split": "test", "hf_repo": "BeIR/scifact"},
        )
        assert bundle.metadata == {
            "source": "beir",
            "split": "test",
            "hf_repo": "BeIR/scifact",
        }

    def test_sample_passages_works_after_load(self) -> None:
        """Regression: int-id sampling broke before bundle.py normalized dtypes."""
        corpus = [{"_id": i, "text": f"doc {i}"} for i in range(20)]
        queries = [{"_id": i, "text": f"q{i}"} for i in range(5)]
        qrels = [{"query-id": i, "corpus-id": i, "score": 1} for i in range(5)]
        bundle = bundle_from_beir_records("toy", corpus, queries, qrels)
        sub = bundle.sample_passages(n=10, seed=0)
        assert sub.n_passages == 10


# ---------------------------------------------------------------------------
# Streaming sampler
# ---------------------------------------------------------------------------


def _make_records(n: int) -> list[BeirRecord]:
    return [BeirRecord(pid=f"p{i:04d}", text=f"doc {i}") for i in range(n)]


class TestStreamSampleCorpus:
    def test_judged_passages_always_included(self) -> None:
        records = _make_records(100)
        judged = {"p0007", "p0019", "p0050", "p0099"}
        result = list(_stream_sample_corpus(iter(records), judged, sample_size=20, seed=0))
        kept_pids = {r.pid for r in result}
        assert judged.issubset(kept_pids)

    def test_total_size_matches_quota(self) -> None:
        records = _make_records(1000)
        judged = {"p0001", "p0002"}
        result = list(_stream_sample_corpus(iter(records), judged, sample_size=50, seed=0))
        assert len(result) == 50

    def test_quota_capped_at_corpus_size(self) -> None:
        records = _make_records(10)
        judged: set[str] = set()
        result = list(_stream_sample_corpus(iter(records), judged, sample_size=999, seed=0))
        assert len(result) == 10

    def test_deterministic_for_same_seed(self) -> None:
        records = _make_records(200)
        judged = {"p0001"}
        a = [r.pid for r in _stream_sample_corpus(iter(records), judged, 30, seed=7)]
        b = [r.pid for r in _stream_sample_corpus(iter(records), judged, 30, seed=7)]
        assert sorted(a) == sorted(b)

    def test_different_seed_diverges(self) -> None:
        records = _make_records(500)
        judged: set[str] = set()
        a = sorted(r.pid for r in _stream_sample_corpus(iter(records), judged, 50, seed=1))
        b = sorted(r.pid for r in _stream_sample_corpus(iter(records), judged, 50, seed=2))
        assert a != b

    def test_order_independent(self) -> None:
        """Reordering the corpus stream must not change the sampled set."""
        records = _make_records(200)
        judged = {"p0050"}
        forward = sorted(r.pid for r in _stream_sample_corpus(iter(records), judged, 30, seed=11))
        reverse = sorted(
            r.pid for r in _stream_sample_corpus(iter(reversed(records)), judged, 30, seed=11)
        )
        assert forward == reverse

    def test_judged_dedup_within_stream(self) -> None:
        """Repeated judged pids in the stream emit at most once."""
        records = _make_records(20) + [BeirRecord(pid="p0001", text="dup")] * 5
        judged = {"p0001"}
        result = list(_stream_sample_corpus(iter(records), judged, 5, seed=0))
        kept = [r for r in result if r.pid == "p0001"]
        assert len(kept) == 1

    def test_sample_size_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            list(_stream_sample_corpus(iter([]), set(), sample_size=0, seed=0))

    def test_orphan_judged_pid_does_not_shrink_result(self) -> None:
        """Quota must size off pids actually present, not off the judged set."""
        records = _make_records(50)
        # Reference 5 phantom pids in `judged` that are NOT in the corpus.
        judged = {f"GHOST{i}" for i in range(5)} | {"p0001"}
        result = list(_stream_sample_corpus(iter(records), judged, sample_size=20, seed=0))
        # 1 real judged ("p0001") + 19 unjudged = 20 total. Without the fix
        # the orphan ghosts would each consume a quota slot and shrink this
        # to 15.
        assert len(result) == 20

    def test_unjudged_duplicate_rows_consume_one_slot(self) -> None:
        """Duplicate unjudged pids must not eat multiple quota slots."""
        # Stream 200 unique unjudged pids, each repeated 3 times.
        records = []
        for rec in _make_records(200):
            records.extend([rec, rec, rec])
        result = list(_stream_sample_corpus(iter(records), set(), sample_size=50, seed=0))
        # Exactly 50 unique pids — duplicates should not have consumed slots.
        kept_pids = {r.pid for r in result}
        assert len(result) == 50
        assert len(kept_pids) == 50

    def test_judged_exceeds_sample_size_truncates_deterministically(self) -> None:
        """If |judged_in_corpus| > sample_size, we keep top-N by priority."""
        records = _make_records(100)
        judged = {r.pid for r in records[:80]}
        a = list(_stream_sample_corpus(iter(records), judged, sample_size=10, seed=3))
        b = list(_stream_sample_corpus(iter(records), judged, sample_size=10, seed=3))
        assert len(a) == 10
        assert {r.pid for r in a} == {r.pid for r in b}


class TestStablePriority:
    def test_deterministic(self) -> None:
        assert _stable_priority("foo", 1) == _stable_priority("foo", 1)

    def test_pid_changes_priority(self) -> None:
        assert _stable_priority("foo", 1) != _stable_priority("bar", 1)

    def test_seed_changes_priority(self) -> None:
        assert _stable_priority("foo", 1) != _stable_priority("foo", 2)
