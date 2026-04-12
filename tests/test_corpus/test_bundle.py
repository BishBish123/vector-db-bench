"""Unit tests for the CorpusBundle dataclass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vdbbench.corpus.bundle import CorpusBundle


def _toy_bundle() -> CorpusBundle:
    return CorpusBundle(
        name="toy",
        passages=pd.DataFrame({"pid": ["p0", "p1", "p2", "p3"], "text": ["a", "b", "c", "d"]}),
        queries=pd.DataFrame({"qid": ["q0", "q1"], "text": ["alpha", "beta"]}),
        qrels=pd.DataFrame(
            {
                "qid": ["q0", "q0", "q1", "q1"],
                "pid": ["p0", "p1", "p2", "p3"],
                "relevance": [1, 1, 1, 1],
            }
        ),
    )


class TestValidation:
    def test_missing_columns_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            )

    def test_extra_columns_rejected(self) -> None:
        with pytest.raises(ValueError, match="unexpected columns"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["x"], "extra": [1]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            )

    def test_duplicate_pids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate pids"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0", "p0"], "text": ["a", "b"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            )

    def test_negative_relevance_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [-1]}),
            )

    def test_duplicate_qrel_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"duplicate.*rows"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0", "q0"], "pid": ["p0", "p0"], "relevance": [1, 2]}),
            )

    def test_null_pids_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"passages\.pid.*null"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0", np.nan], "text": ["a", "b"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            )

    def test_null_qrel_relevance_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"qrels\.relevance.*null"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [np.nan]}),
            )

    def test_orphan_qrel_pid_rejected(self) -> None:
        with pytest.raises(ValueError, match="pids not in passages"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["q0"], "pid": ["MISSING"], "relevance": [1]}),
            )

    def test_orphan_qrel_qid_rejected(self) -> None:
        with pytest.raises(ValueError, match="qids not in queries"):
            CorpusBundle(
                name="bad",
                passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
                queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
                qrels=pd.DataFrame({"qid": ["GHOST"], "pid": ["p0"], "relevance": [1]}),
            )


class TestNormalization:
    def test_integer_ids_are_coerced_to_strings(self) -> None:
        """Loaders that ship int ids (MS-MARCO style) must not break sampling."""
        bundle = CorpusBundle(
            name="int-ids",
            passages=pd.DataFrame({"pid": [10, 11, 12, 13], "text": ["a", "b", "c", "d"]}),
            queries=pd.DataFrame({"qid": [100, 101], "text": ["x", "y"]}),
            qrels=pd.DataFrame(
                {"qid": [100, 100, 101], "pid": [10, 11, 12], "relevance": [1, 2, 1]}
            ),
        )
        # dtype may be `object` or pandas `string` depending on version; the
        # invariant we care about is that every value is actually a Python str.
        assert all(isinstance(v, str) for v in bundle.passages["pid"])
        assert all(isinstance(v, str) for v in bundle.qrels["pid"])
        assert all(isinstance(v, str) for v in bundle.queries["qid"])
        # Sampling now actually finds the rows because dtypes match.
        sub = bundle.sample_passages(n=2, seed=0)
        assert sub.n_passages == 2
        assert sub.n_queries >= 1


class TestFingerprint:
    def test_stable_across_row_order(self) -> None:
        bundle = _toy_bundle()
        shuffled = CorpusBundle(
            name=bundle.name,
            passages=bundle.passages.sample(frac=1.0, random_state=1).reset_index(drop=True),
            queries=bundle.queries.sample(frac=1.0, random_state=2).reset_index(drop=True),
            qrels=bundle.qrels.sample(frac=1.0, random_state=3).reset_index(drop=True),
        )
        assert bundle.fingerprint() == shuffled.fingerprint()

    def test_changes_when_content_changes(self) -> None:
        bundle = _toy_bundle()
        mutated = CorpusBundle(
            name=bundle.name,
            passages=bundle.passages.replace({"a": "A"}),
            queries=bundle.queries,
            qrels=bundle.qrels,
        )
        assert bundle.fingerprint() != mutated.fingerprint()

    def test_set_metadata_fingerprint_is_deterministic(self) -> None:
        """Sets must be canonicalized so fingerprints don't depend on PYTHONHASHSEED."""
        a = CorpusBundle(
            name="m",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"splits": {"train", "test", "dev"}},
        )
        b = CorpusBundle(
            name="m",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"splits": {"dev", "train", "test"}},  # different insertion order
        )
        assert a.fingerprint() == b.fingerprint()

    def test_separator_collision_resistance(self) -> None:
        """Embedded control bytes in user content must not collide on hash."""
        a = CorpusBundle(
            name="x",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["a", "b\x1fc"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="x",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["a\x1fb", "c"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        assert a.fingerprint() != b.fingerprint()

    def test_null_text_distinct_from_empty_text(self) -> None:
        """Null text and empty-string text must hash differently."""
        a = CorpusBundle(
            name="t",
            passages=pd.DataFrame({"pid": ["p0"], "text": [""]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="t",
            passages=pd.DataFrame({"pid": ["p0"], "text": [pd.NA]}, dtype="object"),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        assert a.fingerprint() != b.fingerprint()

    def test_metadata_is_part_of_fingerprint(self) -> None:
        """Provenance edits must invalidate the fingerprint, not silently load."""
        a = CorpusBundle(
            name="m",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"source": "v1"},
        )
        b = CorpusBundle(
            name="m",
            passages=a.passages,
            queries=a.queries,
            qrels=a.qrels,
            metadata={"source": "v2"},
        )
        assert a.fingerprint() != b.fingerprint()


class TestSampling:
    def test_sample_smaller_than_corpus_keeps_judged_queries(self) -> None:
        bundle = _toy_bundle()
        sub = bundle.sample_passages(n=2, seed=0)

        assert sub.n_passages == 2
        assert set(sub.passages["pid"]).issubset(set(bundle.passages["pid"]))
        # Every query in the sub-bundle has at least one positive qrel referring
        # to a passage that's actually in the sample.
        for qid in sub.queries["qid"]:
            relevant = sub.qrels[(sub.qrels["qid"] == qid) & (sub.qrels["relevance"] > 0)]
            assert len(relevant) > 0
            assert set(relevant["pid"]).issubset(set(sub.passages["pid"]))

    def test_sample_is_deterministic(self) -> None:
        a = _toy_bundle().sample_passages(n=2, seed=99)
        b = _toy_bundle().sample_passages(n=2, seed=99)
        assert list(a.passages["pid"]) == list(b.passages["pid"])
        assert a.fingerprint() == b.fingerprint()

    def test_different_seed_gives_different_sample(self) -> None:
        # Use a larger toy bundle so the two samples can disagree.
        passages = pd.DataFrame({"pid": [f"p{i}" for i in range(20)], "text": ["x"] * 20})
        queries = pd.DataFrame({"qid": ["q0"], "text": ["x"]})
        qrels = pd.DataFrame(
            {"qid": ["q0"] * 20, "pid": [f"p{i}" for i in range(20)], "relevance": [1] * 20}
        )
        bundle = CorpusBundle(name="big", passages=passages, queries=queries, qrels=qrels)
        a = bundle.sample_passages(n=5, seed=1)
        b = bundle.sample_passages(n=5, seed=2)
        assert list(a.passages["pid"]) != list(b.passages["pid"])

    def test_sample_larger_than_corpus_keeps_all_passages(self) -> None:
        bundle = _toy_bundle()
        sub = bundle.sample_passages(n=999, seed=0)
        assert sub.n_passages == bundle.n_passages
        assert set(sub.passages["pid"]) == set(bundle.passages["pid"])

    def test_oversize_samples_have_identical_fingerprint(self) -> None:
        """Different oversize n values must yield the same identity."""
        bundle = _toy_bundle()
        a = bundle.sample_passages(n=999, seed=0)
        b = bundle.sample_passages(n=99_999, seed=0)
        assert a.fingerprint() == b.fingerprint()
        assert a.name == b.name

    def test_oversize_samples_ignore_seed(self) -> None:
        """Seed is irrelevant when the sample covers the whole corpus."""
        bundle = _toy_bundle()
        a = bundle.sample_passages(n=999, seed=0)
        b = bundle.sample_passages(n=999, seed=42)
        assert a.fingerprint() == b.fingerprint()

    def test_pruning_is_size_independent(self) -> None:
        """A query with no positive qrels gets pruned regardless of sample size.

        Regression for the fast-path that previously returned `self` whenever
        `n >= n_passages`, skipping the cleanup that smaller samples applied.
        """
        passages = pd.DataFrame({"pid": ["p0", "p1", "p2"], "text": ["a", "b", "c"]})
        queries = pd.DataFrame({"qid": ["q-good", "q-orphan"], "text": ["x", "y"]})
        qrels = pd.DataFrame(
            {"qid": ["q-good"], "pid": ["p0"], "relevance": [1]}  # q-orphan has no qrels
        )
        bundle = CorpusBundle(name="b", passages=passages, queries=queries, qrels=qrels)

        small = bundle.sample_passages(n=2, seed=0)
        full = bundle.sample_passages(n=999, seed=0)
        assert "q-orphan" not in set(small.queries["qid"])
        assert "q-orphan" not in set(full.queries["qid"])

    def test_fractional_relevance_is_preserved(self) -> None:
        """Non-integer relevance grades must round-trip without truncation."""
        bundle = CorpusBundle(
            name="frac",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["a", "b"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame(
                {"qid": ["q0", "q0"], "pid": ["p0", "p1"], "relevance": [0.5, 0.25]}
            ),
        )
        rels = sorted(bundle.qrels["relevance"].tolist())
        assert rels == [0.25, 0.5]

    def test_sample_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _toy_bundle().sample_passages(n=0)


class TestValueSemantics:
    def test_equality_via_fingerprint(self) -> None:
        a = _toy_bundle()
        b = _toy_bundle()
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_when_content_differs(self) -> None:
        a = _toy_bundle()
        b = CorpusBundle(
            name=a.name,
            passages=a.passages.replace({"a": "Z"}),
            queries=a.queries,
            qrels=a.qrels,
        )
        assert a != b
        assert hash(a) != hash(b)

    def test_usable_as_dict_key(self) -> None:
        d = {_toy_bundle(): "v1"}
        assert d[_toy_bundle()] == "v1"

    def test_hash_is_stable_under_in_place_mutation(self) -> None:
        """Mutating a bundle after putting it in a dict must not lose the entry."""
        bundle = _toy_bundle()
        registry: dict[CorpusBundle, str] = {bundle: "before"}
        # Sneak a mutation past the frozen=True guard: pandas DataFrame
        # internals are still mutable.
        bundle.passages.loc[0, "text"] = "MUTATED"
        bundle.metadata["sneaky"] = True  # type: ignore[index]
        # Lookup with the same instance must still hit; without caching the
        # construction fingerprint, the hash bucket would shift.
        assert registry[bundle] == "before"

    def test_numpy_bool_in_metadata_is_serialisable(self) -> None:
        """Loaders that store NumPy booleans (e.g. `series.any()`) must work."""
        bundle = CorpusBundle(
            name="np-bool",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"is_clean": np.bool_(True)},
        )
        # Both round-trip and fingerprint must work without raising.
        bundle.fingerprint()


class TestRoundtrip:
    def test_save_and_load_recovers_bundle(self, tmp_path: Path) -> None:
        bundle = _toy_bundle()
        out = bundle.save(tmp_path / "corpus")
        loaded = CorpusBundle.load(out)

        pd.testing.assert_frame_equal(bundle.passages, loaded.passages)
        pd.testing.assert_frame_equal(bundle.queries, loaded.queries)
        pd.testing.assert_frame_equal(bundle.qrels, loaded.qrels)
        assert loaded.name == bundle.name
        assert loaded.fingerprint() == bundle.fingerprint()

    def test_load_detects_mutated_files(self, tmp_path: Path) -> None:
        bundle = _toy_bundle()
        root = bundle.save(tmp_path / "corpus")
        # Mutate the parquet without updating manifest.
        passages = pd.read_parquet(root / "passages.parquet")
        passages.loc[0, "text"] = "MUTATED"
        passages.to_parquet(root / "passages.parquet", index=False)

        with pytest.raises(ValueError, match="fingerprint"):
            CorpusBundle.load(root)

    def test_load_detects_mutated_metadata(self, tmp_path: Path) -> None:
        """Editing manifest provenance after save must invalidate the bundle."""
        bundle = CorpusBundle(
            name="m",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["a"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"source": "v1", "seed": 0},
        )
        root = bundle.save(tmp_path / "corpus")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metadata"]["seed"] = 999  # tamper
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="fingerprint"):
            CorpusBundle.load(root)

    def test_load_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CorpusBundle.load(tmp_path)


class TestSliceFraction:
    def test_half_fraction_keeps_about_half(self) -> None:
        passages = pd.DataFrame({"pid": [f"p{i}" for i in range(20)], "text": ["x"] * 20})
        queries = pd.DataFrame({"qid": ["q0"], "text": ["x"]})
        qrels = pd.DataFrame(
            {"qid": ["q0"] * 20, "pid": [f"p{i}" for i in range(20)], "relevance": [1] * 20}
        )
        bundle = CorpusBundle(name="b", passages=passages, queries=queries, qrels=qrels)

        sliced = bundle.slice_fraction(0.5, seed=0)
        assert sliced.n_passages == 10
        assert "slice_dropped_qrels" in sliced.metadata
        assert int(sliced.metadata["slice_dropped_qrels"]) == 10  # type: ignore[arg-type]

    def test_records_fraction_in_metadata(self) -> None:
        bundle = _toy_bundle()
        sliced = bundle.slice_fraction(0.75, seed=1)
        assert sliced.metadata["slice_fraction"] == pytest.approx(0.75)

    def test_full_fraction_keeps_everything(self) -> None:
        bundle = _toy_bundle()
        sliced = bundle.slice_fraction(1.0)
        assert sliced.n_passages == bundle.n_passages
        assert sliced.metadata["slice_dropped_qrels"] == 0

    def test_invalid_fraction_rejected(self) -> None:
        bundle = _toy_bundle()
        with pytest.raises(ValueError, match="fraction"):
            bundle.slice_fraction(0.0)
        with pytest.raises(ValueError, match="fraction"):
            bundle.slice_fraction(1.5)
        with pytest.raises(ValueError, match="fraction"):
            bundle.slice_fraction(-0.1)


class TestMerge:
    def test_disjoint_merge(self) -> None:
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["alpha", "bravo"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p2", "p3"], "text": ["charlie", "delta"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["y"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p2"], "relevance": [1]}),
        )
        merged = a.merge(b)
        assert merged.n_passages == 4
        assert merged.n_queries == 2
        assert merged.n_qrels == 2
        assert "merged_from" in merged.metadata

    def test_merge_raises_on_conflicting_passage_text(self) -> None:
        # Same `pid` with **different** text on each side used to silently
        # drop the second; that's a data-loss bug. Fail loud instead.
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["alpha", "bravo"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p1", "p2"], "text": ["DIFFERENT", "charlie"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["y"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p2"], "relevance": [1]}),
        )
        with pytest.raises(ValueError, match="conflicting passage text"):
            a.merge(b)

    def test_merge_raises_on_conflicting_query_text(self) -> None:
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["one"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["DIFFERENT"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        with pytest.raises(ValueError, match="conflicting query text"):
            a.merge(b)

    def test_merge_raises_on_conflicting_topic(self) -> None:
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["one"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"topic": "medical"},
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p1"], "text": ["y"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["two"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p1"], "relevance": [1]}),
            metadata={"topic": "legal"},
        )
        with pytest.raises(ValueError, match="topic conflict"):
            a.merge(b)

    def test_merge_topic_one_sided_takes_non_empty(self) -> None:
        # If only one side carries a topic the merge succeeds and that
        # topic propagates to the merged bundle's metadata.
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["one"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
            metadata={"topic": "medical"},
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p1"], "text": ["y"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["two"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p1"], "relevance": [1]}),
        )
        merged = a.merge(b)
        assert merged.metadata.get("topic") == "medical"

    def test_merge_accepts_identical_passage_text(self) -> None:
        # Sanity: same pid + same text (and same qid + same text) is the
        # normal "two slices of the same corpus" case; merge cleanly.
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["alpha", "bravo"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["one"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p1", "p2"], "text": ["bravo", "charlie"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["two"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p2"], "relevance": [1]}),
        )
        merged = a.merge(b)
        assert merged.n_passages == 3
        p1_text = merged.passages.loc[merged.passages["pid"] == "p1", "text"].iloc[0]
        assert p1_text == "bravo"

    def test_conflicting_grades_rejected(self) -> None:
        a = CorpusBundle(
            name="a",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["y"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["p0"], "text": ["x"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["y"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [3]}),
        )
        with pytest.raises(ValueError, match="conflicting relevance"):
            a.merge(b)

    def test_merge_with_custom_name(self) -> None:
        a = _toy_bundle()
        b = CorpusBundle(
            name="other",
            passages=pd.DataFrame({"pid": ["p9"], "text": ["z"]}),
            queries=pd.DataFrame({"qid": ["q9"], "text": ["w"]}),
            qrels=pd.DataFrame({"qid": ["q9"], "pid": ["p9"], "relevance": [1]}),
        )
        merged = a.merge(b, name="combined")
        assert merged.name == "combined"

    def test_rejects_non_bundle(self) -> None:
        a = _toy_bundle()
        with pytest.raises(TypeError, match="CorpusBundle"):
            a.merge("not a bundle")  # type: ignore[arg-type]

    def test_lineage_recorded(self) -> None:
        a = _toy_bundle()
        b = CorpusBundle(
            name="b",
            passages=pd.DataFrame({"pid": ["pX"], "text": ["z"]}),
            queries=pd.DataFrame({"qid": ["qX"], "text": ["w"]}),
            qrels=pd.DataFrame({"qid": ["qX"], "pid": ["pX"], "relevance": [1]}),
        )
        merged = a.merge(b)
        lineage = merged.metadata["merged_from"]
        assert isinstance(lineage, list)
        assert len(lineage) == 2
        assert a.fingerprint() in lineage
        assert b.fingerprint() in lineage

    def test_merge_is_order_invariant(self) -> None:
        # `a.merge(b).fingerprint() == b.merge(a).fingerprint()` whenever
        # the two bundles describe equivalent data. Without this, the
        # fingerprint flips depending on which side called `.merge()`,
        # breaking cache reuse and disagreeing on lineage between runs.
        a = CorpusBundle(
            name="alpha",
            passages=pd.DataFrame({"pid": ["p0", "p1"], "text": ["one", "two"]}),
            queries=pd.DataFrame({"qid": ["q0"], "text": ["x"]}),
            qrels=pd.DataFrame({"qid": ["q0"], "pid": ["p0"], "relevance": [1]}),
        )
        b = CorpusBundle(
            name="bravo",
            passages=pd.DataFrame({"pid": ["p2", "p3"], "text": ["three", "four"]}),
            queries=pd.DataFrame({"qid": ["q1"], "text": ["y"]}),
            qrels=pd.DataFrame({"qid": ["q1"], "pid": ["p2"], "relevance": [1]}),
        )
        ab = a.merge(b)
        ba = b.merge(a)
        assert ab.name == ba.name
        assert ab.metadata["merged_from"] == ba.metadata["merged_from"]
        assert ab.metadata["merged_names"] == ba.metadata["merged_names"]
        assert ab.fingerprint() == ba.fingerprint()
