"""Corpus loaders + ground-truth handling.

Every loader returns a `CorpusBundle` containing passages, queries and qrels
in a uniform shape so the rest of the benchmark stays loader-agnostic.
"""

from vdbbench.corpus.beir import (
    BeirQrel,
    BeirQuery,
    BeirRecord,
    bundle_from_beir_records,
    load_beir_dataset,
    load_msmarco,
)
from vdbbench.corpus.bundle import CorpusBundle
from vdbbench.corpus.synthetic import SyntheticConfig, generate_synthetic

__all__ = [
    "BeirQrel",
    "BeirQuery",
    "BeirRecord",
    "CorpusBundle",
    "SyntheticConfig",
    "bundle_from_beir_records",
    "generate_synthetic",
    "load_beir_dataset",
    "load_msmarco",
]
