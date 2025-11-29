"""Encoder protocol + corpus encoding helpers + serialization.

The actual sentence-transformer wrapper lives in `sentence_transformer.py` so
this module stays importable without `torch` installed (Intel macOS), and
tests can use the in-package `FakeEncoder` for shape/integration coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vdbbench.corpus.bundle import CorpusBundle


@runtime_checkable
class Encoder(Protocol):
    """A text encoder producing dense float32 vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode `texts` into an `(N, dim)` float32 ndarray.

        Implementations must be deterministic given the same input + state
        and must preserve input order (encode `texts[i]` to row `i`).
        """
        ...


def detect_device() -> str:
    """Return `"cuda"`, `"mps"`, or `"cpu"` — whichever this process can use.

    Falls back to `"cpu"` if torch is not importable, which lets non-ML
    callers detect the device without taking a hard dep on torch.
    """
    try:
        import torch  # noqa: PLC0415  -- optional import
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# FakeEncoder — deterministic, no-torch encoder for tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeEncoder:
    """Hash-based encoder so tests can exercise the encoding pipeline with
    no model download and no torch installed.

    Vectors are deterministic per input text: `vec[i] = H(text)[:dim]`
    interpreted as float32 then L2-normalized.
    """

    name: str = "fake-test-encoder"
    dim: int = 32

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.dim > 64:
            raise ValueError("FakeEncoder dim is capped at 64 for hash-based determinism")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for j, text in enumerate(chunk):
                digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
                # Pull `dim` bytes, normalize to [-1, 1).
                raw = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32)
                vec = raw / 127.5 - 1.0
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                out[i + j] = vec
        return out


# ---------------------------------------------------------------------------
# EncodedBundle — corpus + dense vectors for both passages and queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EncodedBundle:
    """Pairs a CorpusBundle with its dense vector representation.

    Equality and hashing are disabled — the bundle has its own fingerprint
    and the matrices are unhashable. Compare via `bundle.fingerprint()` and
    `np.allclose` on the matrices when needed.
    """

    bundle: CorpusBundle
    passage_vectors: np.ndarray  # (n_passages, dim) float32, row-aligned with bundle.passages
    query_vectors: np.ndarray  # (n_queries, dim)   float32, row-aligned with bundle.queries
    encoder_name: str
    metadata: dict[str, object] = field(default_factory=dict)
    # Snapshot the bundle's fingerprint at construction so that in-place
    # mutations of `bundle.passages` between `encode_corpus()` and `save()`
    # are caught instead of silently writing stale vectors against the
    # mutated corpus.
    _encoded_at_fingerprint: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        # Shape + dtype invariants — adapters and metric code rely on these.
        if self.passage_vectors.ndim != 2:
            raise ValueError(
                f"passage_vectors must be 2-D, got shape {self.passage_vectors.shape!r}"
            )
        if self.query_vectors.ndim != 2:
            raise ValueError(f"query_vectors must be 2-D, got shape {self.query_vectors.shape!r}")
        if self.passage_vectors.shape[0] != self.bundle.n_passages:
            raise ValueError(
                f"passage_vectors row count {self.passage_vectors.shape[0]} != "
                f"bundle.n_passages {self.bundle.n_passages}"
            )
        if self.query_vectors.shape[0] != self.bundle.n_queries:
            raise ValueError(
                f"query_vectors row count {self.query_vectors.shape[0]} != "
                f"bundle.n_queries {self.bundle.n_queries}"
            )
        if self.passage_vectors.shape[1] != self.query_vectors.shape[1]:
            raise ValueError(
                f"passage and query vector dims disagree: "
                f"{self.passage_vectors.shape[1]} vs {self.query_vectors.shape[1]}"
            )
        # Force float32 so adapters never have to widen.
        if self.passage_vectors.dtype != np.float32:
            object.__setattr__(self, "passage_vectors", self.passage_vectors.astype(np.float32))
        if self.query_vectors.dtype != np.float32:
            object.__setattr__(self, "query_vectors", self.query_vectors.astype(np.float32))
        # Pin the bundle fingerprint as it was when these vectors were minted.
        object.__setattr__(self, "_encoded_at_fingerprint", self.bundle.fingerprint())

    @property
    def dim(self) -> int:
        return int(self.passage_vectors.shape[1])

    # ---------- IO ----------

    def save(self, root: str | Path) -> Path:
        """Persist the bundle (parquet + manifest) plus vectors as parquet.

        Vectors are stored as fixed-size lists of float32 in parquet so
        they round-trip losslessly and re-open in any pyarrow-aware reader.
        Refuses to save if the bundle was mutated in place after encoding —
        the saved corpus would otherwise carry the new ids/text while the
        on-disk vectors still reflect the old ones.
        """
        if self.bundle.fingerprint() != self._encoded_at_fingerprint:
            raise RuntimeError(
                "CorpusBundle was mutated in place after encoding; the saved "
                "vectors would no longer match `corpus/`. Re-encode before saving."
            )
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.bundle.save(root_path / "corpus")

        _write_vectors(
            self.passage_vectors, self.bundle.passages["pid"], root_path / "passages.parquet"
        )
        _write_vectors(
            self.query_vectors, self.bundle.queries["qid"], root_path / "queries.parquet"
        )

        # Pulled from `vdbbench.corpus.bundle` so manifest values (Paths, NumPy
        # scalars, sets) survive `json.dumps` the same way the corpus
        # manifest does.
        from vdbbench.corpus.bundle import _jsonable  # noqa: PLC0415

        manifest = {
            "encoder_name": self.encoder_name,
            "dim": self.dim,
            "n_passages": int(self.bundle.n_passages),
            "n_queries": int(self.bundle.n_queries),
            # Bind the encoded artifact to its source bundle so swapping
            # `corpus/` with a different bundle that happens to reuse the
            # same ids does not silently reattach stale vectors.
            "bundle_fingerprint": self.bundle.fingerprint(),
            "metadata": _jsonable(self.metadata),
        }
        (root_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return root_path


def load_encoded_bundle(root: str | Path) -> EncodedBundle:
    """Load an `EncodedBundle` previously written by `save()`."""
    root_path = Path(root)
    bundle = CorpusBundle.load(root_path / "corpus")
    manifest = json.loads((root_path / "manifest.json").read_text())

    expected_fp = manifest.get("bundle_fingerprint")
    if expected_fp and expected_fp != bundle.fingerprint():
        raise ValueError(
            "encoded bundle's stored corpus fingerprint does not match the "
            "loaded corpus — somebody swapped `corpus/` under the saved "
            "vectors. Refusing to load (would silently mis-pair embeddings)."
        )

    expected_dim = int(manifest["dim"])
    passage_vecs = _read_vectors(
        root_path / "passages.parquet", bundle.passages["pid"], expected_dim
    )
    query_vecs = _read_vectors(root_path / "queries.parquet", bundle.queries["qid"], expected_dim)

    return EncodedBundle(
        bundle=bundle,
        passage_vectors=passage_vecs,
        query_vectors=query_vecs,
        encoder_name=str(manifest["encoder_name"]),
        metadata=dict(manifest.get("metadata", {})),
    )


def _write_vectors(vecs: np.ndarray, ids: pd.Series[str], path: Path) -> None:
    """Write (id, vector) parquet without materializing the matrix as Python.

    `pa.FixedSizeListArray.from_arrays` reuses the contiguous numpy buffer
    rather than going through `vecs.tolist()`, which on a 100k by 384 matrix
    would otherwise allocate ~40M Python float objects (~1 GB).
    """
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32, copy=False)
    if not vecs.flags["C_CONTIGUOUS"]:
        vecs = np.ascontiguousarray(vecs)
    dim = int(vecs.shape[1])
    flat = pa.array(vecs.reshape(-1), type=pa.float32())
    vector_col = pa.FixedSizeListArray.from_arrays(flat, dim)
    table = pa.table({"id": pa.array(list(ids), type=pa.string()), "vector": vector_col})
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _read_vectors(path: Path, expected_ids: pd.Series[str], expected_dim: int) -> np.ndarray:
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    ids = table.column("id").to_pylist()
    if list(expected_ids) != ids:
        raise ValueError(
            f"vector file {path.name} ids do not match the bundle "
            f"(possibly mutated; expected {len(expected_ids)} ids, got {len(ids)})"
        )
    if not ids:
        # `np.array([])` would be 1-D, breaking the EncodedBundle invariant.
        return np.zeros((0, expected_dim), dtype=np.float32)
    arr = np.array(table.column("vector").to_pylist(), dtype=np.float32)
    actual_dim = int(arr.shape[1]) if arr.ndim == 2 else -1
    if actual_dim != expected_dim:
        raise ValueError(
            f"vector file {path.name} has dim {actual_dim} but the manifest "
            f"declares dim {expected_dim} — likely a different encoder's "
            f"output got swapped in"
        )
    return arr


# ---------------------------------------------------------------------------
# Corpus encoding pipeline
# ---------------------------------------------------------------------------


def encode_corpus(
    bundle: CorpusBundle,
    encoder: Encoder,
    batch_size: int = 64,
    metadata: dict[str, object] | None = None,
) -> EncodedBundle:
    """Encode every passage and every query in `bundle` with `encoder`.

    Pure orchestrator: doesn't know about `torch` or any specific encoder
    backend. Pass any object satisfying the `Encoder` protocol.

    Row alignment is preserved: `passage_vectors[i]` is the embedding of
    `bundle.passages.iloc[i]["text"]`, and analogously for queries.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    passage_texts = bundle.passages["text"].astype(str).tolist()
    query_texts = bundle.queries["text"].astype(str).tolist()

    passage_vecs = encoder.encode(passage_texts, batch_size=batch_size)
    query_vecs = encoder.encode(query_texts, batch_size=batch_size)

    if passage_vecs.shape[0] != len(passage_texts):
        raise RuntimeError(
            f"encoder returned {passage_vecs.shape[0]} passage vectors but "
            f"received {len(passage_texts)} texts — row alignment broken"
        )
    if query_vecs.shape[0] != len(query_texts):
        raise RuntimeError(
            f"encoder returned {query_vecs.shape[0]} query vectors but "
            f"received {len(query_texts)} texts — row alignment broken"
        )

    return EncodedBundle(
        bundle=bundle,
        passage_vectors=passage_vecs,
        query_vectors=query_vecs,
        encoder_name=encoder.name,
        metadata={
            "encoder_dim": encoder.dim,
            "batch_size": batch_size,
            **(metadata or {}),
        },
    )
