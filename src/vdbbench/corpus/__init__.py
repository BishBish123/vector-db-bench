"""Corpus loaders + ground-truth handling.

Every loader returns a `CorpusBundle` containing passages, queries and qrels
in a uniform shape so the rest of the benchmark stays loader-agnostic.
"""

from vdbbench.corpus.bundle import CorpusBundle
from vdbbench.corpus.synthetic import SyntheticConfig, generate_synthetic

__all__ = [
    "CorpusBundle",
    "SyntheticConfig",
    "generate_synthetic",
]
