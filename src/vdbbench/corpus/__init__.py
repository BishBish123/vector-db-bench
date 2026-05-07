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
from vdbbench.corpus.wikipedia import (
    CorpusChunk,
    CorpusDoc,
    WikipediaCorpus,
    iter_chunks,
)

__all__ = [
    "BeirQrel",
    "BeirQuery",
    "BeirRecord",
    "CorpusBundle",
    "CorpusChunk",
    "CorpusDoc",
    "SyntheticConfig",
    "WikipediaCorpus",
    "bundle_from_beir_records",
    "generate_synthetic",
    "iter_chunks",
    "load_beir_dataset",
    "load_msmarco",
]
