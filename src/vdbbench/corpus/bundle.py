"""The shared corpus container used by every loader and adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class CorpusBundle:
    """A passage corpus, the queries against it, and the relevance judgements."""

    name: str
    passages: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
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

    @property
    def n_passages(self) -> int:
        return len(self.passages)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_qrels(self) -> int:
        return len(self.qrels)
