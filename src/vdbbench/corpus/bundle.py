"""The shared corpus container used by every loader and adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

# Canonical column orderings — checked by the validators so loaders can't drift.
PASSAGE_COLS: tuple[str, ...] = ("pid", "text")
QUERY_COLS: tuple[str, ...] = ("qid", "text")
QREL_COLS: tuple[str, ...] = ("qid", "pid", "relevance")


def _ensure_columns(df: pd.DataFrame, expected: tuple[str, ...], name: str) -> pd.DataFrame:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name} dataframe missing columns: {missing}")
    extra = [c for c in df.columns if c not in expected]
    if extra:
        raise ValueError(f"{name} dataframe has unexpected columns: {extra}")
    return df.loc[:, list(expected)].reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class CorpusBundle:
    """A passage corpus, the queries against it, and the relevance judgements.

    `pid` and `qid` are stored as strings so loaders are free to use any id
    scheme (BeIR uses string ids, MS-MARCO uses integers — we normalize).
    `relevance` is non-negative; 0 means "judged but not relevant", >0 means
    relevant (BEIR uses graded relevance for some datasets).
    """

    name: str
    passages: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
    # Filled in __post_init__ from `fingerprint()` after validation. Used by
    # __eq__/__hash__ so identity stays stable across in-place mutations of
    # the underlying frames or metadata.
    _construction_fingerprint: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        # Frozen dataclass means we have to use object.__setattr__ to normalize.
        for attr, cols, label in (
            ("passages", PASSAGE_COLS, "passages"),
            ("queries", QUERY_COLS, "queries"),
            ("qrels", QREL_COLS, "qrels"),
        ):
            df = _ensure_columns(getattr(self, attr), cols, label)
            object.__setattr__(self, attr, df)

        # Reject nulls in id columns before coercion — `.astype(str)` keeps
        # NaN as NaN (not the literal string "nan"), and downstream `.isin()`
        # filters would silently drop those rows during sampling.
        if self.passages["pid"].isna().any():
            raise ValueError("passages.pid contains nulls")
        if self.queries["qid"].isna().any():
            raise ValueError("queries.qid contains nulls")
        if self.qrels["pid"].isna().any() or self.qrels["qid"].isna().any():
            raise ValueError("qrels.pid/qid contains nulls")
        if self.qrels["relevance"].isna().any():
            raise ValueError("qrels.relevance contains nulls")

        # Force canonical dtypes — loaders may pass int ids (MS-MARCO) or
        # mixed types from joins. Without this, downstream `.isin()` filters
        # silently miss rows when sampled ids and corpus ids disagree on dtype.
        # `relevance` is float so graded labels (BEIR ints, normalized 0..1)
        # both round-trip without truncation.
        self.passages["pid"] = self.passages["pid"].astype(str)
        self.queries["qid"] = self.queries["qid"].astype(str)
        self.qrels["qid"] = self.qrels["qid"].astype(str)
        self.qrels["pid"] = self.qrels["pid"].astype(str)
        self.qrels["relevance"] = self.qrels["relevance"].astype("float64")

        if self.passages["pid"].duplicated().any():
            raise ValueError("passages contain duplicate pids")
        if self.queries["qid"].duplicated().any():
            raise ValueError("queries contain duplicate qids")
        if (self.qrels["relevance"] < 0).any():
            raise ValueError("qrels.relevance must be non-negative")

        # Reject duplicate (qid, pid) rows — a loader join could double-count
        # positives or report conflicting grades for the same pair, and either
        # silently corrupts recall/NDCG.
        if self.qrels.duplicated(subset=["qid", "pid"]).any():
            n_dups = int(self.qrels.duplicated(subset=["qid", "pid"]).sum())
            raise ValueError(f"qrels contain {n_dups} duplicate (qid, pid) rows")

        # Reject orphan qrels — they would silently corrupt recall numbers by
        # adding positives no adapter can ever retrieve, or by referencing
        # queries that are not part of the bundle.
        passage_ids = set(self.passages["pid"])
        query_ids = set(self.queries["qid"])
        orphan_pids = set(self.qrels["pid"]) - passage_ids
        if orphan_pids:
            raise ValueError(
                f"qrels reference {len(orphan_pids)} pids not in passages "
                f"(first: {sorted(orphan_pids)[:3]})"
            )
        orphan_qids = set(self.qrels["qid"]) - query_ids
        if orphan_qids:
            raise ValueError(
                f"qrels reference {len(orphan_qids)} qids not in queries "
                f"(first: {sorted(orphan_qids)[:3]})"
            )

        # Cache fingerprint last (after validation passes) so identity is
        # frozen even if callers mutate the dataframes in place later.
        object.__setattr__(self, "_construction_fingerprint", self.fingerprint())

    # ---------- value semantics ----------
    #
    # Auto-generated dataclass __eq__/__hash__ are disabled because pandas
    # DataFrame equality returns a frame (not a bool) and DataFrames are
    # unhashable. Both __eq__ and __hash__ go through a fingerprint that is
    # captured at construction time and stored on the instance, so callers
    # can safely use bundles as dict/set keys: in-place mutation of the
    # underlying DataFrames or metadata after construction will not silently
    # shift the hash bucket and leak entries (the bundle is logically
    # immutable for identity purposes).

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CorpusBundle):
            return NotImplemented
        return self._construction_fingerprint == other._construction_fingerprint

    def __hash__(self) -> int:
        return hash(self._construction_fingerprint)

    # ---------- shape helpers ----------

    @property
    def n_passages(self) -> int:
        return len(self.passages)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_qrels(self) -> int:
        return len(self.qrels)

    # ---------- determinism / fingerprinting ----------

    def fingerprint(self) -> str:
        """A stable hex digest of the bundle's content + provenance metadata.

        Used to detect cache hits and to tag run outputs. Order-invariant
        across the row dimension so different shuffles of the same data hash
        to the same value. Includes `metadata` so post-save edits to the
        manifest are caught at load time.
        """
        h = hashlib.blake2b(digest_size=16)
        for df, cols in (
            (self.passages, PASSAGE_COLS),
            (self.queries, QUERY_COLS),
            (self.qrels, QREL_COLS),
        ):
            ordered = df.sort_values(by=list(cols)).reset_index(drop=True)
            for col in cols:
                # Each value is length-prefixed so embedded delimiters in user
                # content (rare but possible — e.g. `pid='b\x1f\x01c'`) cannot
                # collide with our framing. Nulls get a sentinel length of -1
                # to stay distinguishable from empty strings.
                series = ordered[col].astype("string")
                for value, is_null in zip(series, series.isna(), strict=True):
                    if is_null:
                        h.update(b"\xff\xff\xff\xff")  # u32 sentinel for null
                        continue
                    encoded = str(value).encode("utf-8")
                    h.update(len(encoded).to_bytes(4, "big", signed=False))
                    h.update(encoded)
                h.update(b"||COL||")
            h.update(b"||DF||")
        h.update(self.name.encode("utf-8"))
        h.update(b"\x1d")
        # `metadata` is fingerprinted as canonical JSON so `manifest.json`
        # mutations are caught at load time too.
        h.update(json.dumps(_jsonable(self.metadata), sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    # ---------- sampling ----------

    def sample_passages(
        self,
        n: int,
        seed: int = 42,
        keep_only_judged_queries: bool = True,
    ) -> Self:
        """Return a deterministic sub-bundle restricted to `n` passages.

        Qrels referring to passages not in the sample are dropped. By default,
        queries that lose all of their relevant passages are also dropped so
        downstream recall computations are well-defined. The pruning behavior
        applies regardless of whether the sample is smaller than the corpus.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        rng = np.random.default_rng(seed)

        if n >= self.n_passages:
            new_passages = self.passages.sort_values("pid").reset_index(drop=True)
            new_qrels = self.qrels.sort_values(["qid", "pid"]).reset_index(drop=True)
        else:
            pids_sorted = np.sort(self.passages["pid"].to_numpy().astype(str))
            chosen = rng.choice(pids_sorted, size=n, replace=False)
            chosen.sort()
            keep = pd.Index(chosen)
            new_passages = (
                self.passages[self.passages["pid"].isin(keep)]
                .sort_values("pid")
                .reset_index(drop=True)
            )
            new_qrels = (
                self.qrels[self.qrels["pid"].isin(keep)]
                .sort_values(["qid", "pid"])
                .reset_index(drop=True)
            )

        if keep_only_judged_queries:
            judged = new_qrels[new_qrels["relevance"] > 0]["qid"].unique()
            new_queries = (
                self.queries[self.queries["qid"].isin(judged)]
                .sort_values("qid")
                .reset_index(drop=True)
            )
            new_qrels = new_qrels[new_qrels["qid"].isin(new_queries["qid"])].reset_index(drop=True)
        else:
            new_queries = self.queries.copy().reset_index(drop=True)

        # Clamp `n` for naming/metadata so two oversize requests on the same
        # corpus (n=999, n=10000 against a 4-row corpus) produce identical
        # bundles and identical fingerprints. Seed is also stripped from the
        # identity when the sample is a no-op (n covers the whole corpus),
        # since two oversized calls with different seeds yield the same data.
        effective_n = min(int(n), self.n_passages)
        is_full_corpus = effective_n == self.n_passages
        effective_seed = 0 if is_full_corpus else int(seed)
        new_meta = {
            **self.metadata,
            "sampled_from": self.fingerprint(),
            "sample_n": effective_n,
            "sample_seed": effective_seed,
        }
        return type(self)(
            name=f"{self.name}@n={effective_n}.seed={effective_seed}",
            passages=new_passages,
            queries=new_queries,
            qrels=new_qrels,
            metadata=new_meta,
        )

    # ---------- transformations ----------

    def slice_fraction(
        self,
        fraction: float,
        seed: int = 42,
        keep_only_judged_queries: bool = True,
    ) -> Self:
        """Return a sub-bundle with `round(fraction * n_passages)` passages.

        Convenience over `sample_passages(n=...)` for fraction-of-corpus
        slices ("give me 10 % of the data"). `fraction` is clamped to
        `(0, 1]`; values outside that range raise. Qrels referencing
        passages dropped by the slice are removed and the dropped count
        is recorded in the resulting bundle's metadata under
        `slice_dropped_qrels` so downstream code can surface it.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        n = max(1, round(fraction * self.n_passages))
        sliced = self.sample_passages(
            n=n, seed=seed, keep_only_judged_queries=keep_only_judged_queries
        )
        dropped = self.n_qrels - sliced.n_qrels
        new_meta = {
            **sliced.metadata,
            "slice_fraction": float(fraction),
            "slice_dropped_qrels": int(dropped),
        }
        return type(self)(
            name=sliced.name,
            passages=sliced.passages,
            queries=sliced.queries,
            qrels=sliced.qrels,
            metadata=new_meta,
        )

    def merge(self, other: Self, name: str | None = None) -> Self:
        """Combine `self` and `other` into a single bundle.

        Passage and query ids are unioned. If both bundles carry a row for
        the same `pid` (or `qid`) with **different** text, that's a
        conflict and we raise rather than silently dropping one — the same
        invariant we apply to qrel grade conflicts. If the text matches
        exactly, dedup keeps the first occurrence. Topic conflicts are
        handled the same way: equal or one-sided-empty topics merge
        cleanly; non-empty disagreement raises.

        The result's metadata gets a `merged_from` key with both source
        fingerprints so the lineage is recoverable.
        """
        if not isinstance(other, CorpusBundle):
            raise TypeError(f"merge expects a CorpusBundle, got {type(other)!r}")

        # ---- passage text-conflict check ----
        passage_concat = pd.concat([self.passages, other.passages], ignore_index=True)
        p_conflicts = (
            passage_concat.groupby("pid")["text"].nunique(dropna=False).reset_index(name="n")
        )
        bad_pids = p_conflicts.loc[p_conflicts["n"] > 1, "pid"].tolist()
        if bad_pids:
            raise ValueError(
                f"merge: {len(bad_pids)} pid(s) have conflicting passage text across "
                f"the two bundles (first: {sorted(bad_pids)[:3]})"
            )
        passages = passage_concat.drop_duplicates(subset=["pid"], keep="first").reset_index(
            drop=True
        )

        # ---- query text-conflict check ----
        query_concat = pd.concat([self.queries, other.queries], ignore_index=True)
        q_conflicts = (
            query_concat.groupby("qid")["text"].nunique(dropna=False).reset_index(name="n")
        )
        bad_qids = q_conflicts.loc[q_conflicts["n"] > 1, "qid"].tolist()
        if bad_qids:
            raise ValueError(
                f"merge: {len(bad_qids)} qid(s) have conflicting query text across "
                f"the two bundles (first: {sorted(bad_qids)[:3]})"
            )
        queries = query_concat.drop_duplicates(subset=["qid"], keep="first").reset_index(drop=True)

        qrels_combined = pd.concat([self.qrels, other.qrels], ignore_index=True)
        # Detect conflicting grades for the same (qid, pid) pair before
        # dedup — we'd silently pick whichever row appeared first otherwise.
        conflicts = qrels_combined.groupby(["qid", "pid"])["relevance"].nunique().reset_index()
        bad = conflicts[conflicts["relevance"] > 1]
        if not bad.empty:
            sample = bad.head(3).to_dict(orient="records")
            raise ValueError(
                f"merge: {len(bad)} (qid, pid) pairs have conflicting relevance "
                f"grades across the two bundles (first: {sample})"
            )
        qrels = qrels_combined.drop_duplicates(subset=["qid", "pid"], keep="first").reset_index(
            drop=True
        )

        # ---- topic conflict (metadata-level) ----
        # Topics are tracked under `metadata["topic"]`. If both sides carry
        # a non-empty topic and they disagree, refuse the merge. If one
        # side is empty/missing, the non-empty value wins.
        self_topic = str(self.metadata.get("topic") or "")
        other_topic = str(other.metadata.get("topic") or "")
        if self_topic and other_topic and self_topic != other_topic:
            raise ValueError(
                f"merge: topic conflict — {self_topic!r} (self) vs {other_topic!r} (other)"
            )
        merged_topic = self_topic or other_topic

        merged_name = name if name is not None else f"{self.name}+{other.name}"
        merged_metadata: dict[str, object] = {
            "merged_from": [self.fingerprint(), other.fingerprint()],
            "merged_names": [self.name, other.name],
        }
        if merged_topic:
            merged_metadata["topic"] = merged_topic
        return type(self)(
            name=merged_name,
            passages=passages,
            queries=queries,
            qrels=qrels,
            metadata=merged_metadata,
        )

    # ---------- IO ----------

    def save(self, root: str | Path) -> Path:
        """Persist as a directory of parquet files + a manifest."""
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.passages.to_parquet(root_path / "passages.parquet", index=False)
        self.queries.to_parquet(root_path / "queries.parquet", index=False)
        self.qrels.to_parquet(root_path / "qrels.parquet", index=False)
        manifest = {
            "name": self.name,
            "n_passages": self.n_passages,
            "n_queries": self.n_queries,
            "n_qrels": self.n_qrels,
            "fingerprint": self.fingerprint(),
            "metadata": _jsonable(self.metadata),
        }
        (root_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return root_path

    @classmethod
    def load(cls, root: str | Path) -> Self:
        root_path = Path(root)
        manifest_path = root_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json under {root_path}")
        manifest = json.loads(manifest_path.read_text())
        bundle = cls(
            name=str(manifest["name"]),
            passages=pd.read_parquet(root_path / "passages.parquet"),
            queries=pd.read_parquet(root_path / "queries.parquet"),
            qrels=pd.read_parquet(root_path / "qrels.parquet"),
            metadata=manifest.get("metadata", {}),
        )
        if (fp := manifest.get("fingerprint")) and fp != bundle.fingerprint():
            raise ValueError(
                "loaded bundle fingerprint does not match manifest — corpus has been mutated"
            )
        return bundle


def _jsonable(obj: object) -> object:  # noqa: PLR0911
    """Coerce metadata into something json.dumps will accept.

    Sets are converted to *sorted* lists so that fingerprints stay
    deterministic across processes (Python set iteration order is
    hash-seed dependent). NumPy scalars are unwrapped because they would
    otherwise raise `TypeError: Object of type ... is not JSON serializable`.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, set | frozenset):
        return sorted((_jsonable(v) for v in obj), key=repr)
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
