"""The shared corpus container used by every loader and adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

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

        # Force canonical dtypes — loaders may pass int ids (MS-MARCO).
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

    @property
    def n_passages(self) -> int:
        return len(self.passages)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_qrels(self) -> int:
        return len(self.qrels)
