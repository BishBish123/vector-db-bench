"""Sentence-transformers backed `Encoder` implementation.

Lives in its own module so the rest of the package stays importable without
`torch` installed (Intel macOS doesn't ship torch wheels above 2.2). Tests
that don't need a real model use `vdbbench.embed.encoder.FakeEncoder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vdbbench.embed.encoder import detect_device

if TYPE_CHECKING:  # pragma: no cover - imports only resolved when torch is installed
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    """Wrap a `sentence-transformers` model behind the `Encoder` protocol.

    `device=None` auto-detects (CUDA → MPS → CPU). `normalize=True` matches
    the bge-* / nomic-* convention so adapters can use cosine similarity by
    inner product without re-normalizing.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import (  # noqa: PLC0415  -- optional import
                SentenceTransformer,
            )
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder; "
                "install via `uv sync --extra embed` or use the Docker image."
            ) from e

        self._model_name = model_name
        self._device = device or detect_device()
        self._normalize = normalize
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)
