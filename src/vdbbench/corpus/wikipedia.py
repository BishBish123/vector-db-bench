"""Wikipedia corpus loader + chunker.

Provides two loading paths:
* `WikipediaCorpus.from_huggingface()` — streams the Wikipedia dump via the
  `datasets` library (optional dep, gated behind ``[wiki]`` extras).
* `WikipediaCorpus.from_fixtures()` — reads a JSONL file with
  ``{"id": ..., "title": ..., "text": ...}`` rows. Used in tests and
  offline smoke runs so the full 1M-article download is never required.

Articles are chunked into ``CorpusChunk`` objects via ``iter_chunks()``
before embedding because Wikipedia articles can be many kilobytes — the
chunker is paragraph-aware (splits on ``\\n\\n`` first) and falls back to
hard truncation for paragraphs that still exceed ``chunk_size``.

Typical usage::

    from vdbbench.corpus.wikipedia import WikipediaCorpus, iter_chunks

    docs = WikipediaCorpus.from_fixtures(Path("evals/wikipedia_fixtures.jsonl"))
    chunks = list(iter_chunks(docs, chunk_size=512, overlap=64))
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDoc:
    """A single raw document (Wikipedia article or fixture row).

    Fields mirror the Wikipedia HuggingFace dataset schema and the JSONL
    fixture format so the two loading paths are interchangeable.
    """

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class CorpusChunk:
    """A chunk derived from a ``CorpusDoc``.

    ``chunk_id`` is ``<doc_id>_<chunk_index>`` so the source article is
    recoverable from any chunk. ``title`` is propagated from the parent
    article for re-ranking and display purposes.
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class WikipediaCorpus:
    """Factory-only class: use ``from_huggingface()`` or ``from_fixtures()``."""

    @staticmethod
    def from_huggingface(
        dataset_id: str = "wikipedia",
        config: str = "20231101.en",
        split: str = "train",
        limit: int | None = None,
    ) -> Iterator[CorpusDoc]:
        """Stream Wikipedia articles via the ``datasets`` library.

        Parameters
        ----------
        dataset_id:
            HuggingFace dataset repo id. Defaults to ``"wikipedia"``.
        config:
            Dataset configuration (language + dump date). Defaults to
            ``"20231101.en"`` (English, November 2023 dump).
        split:
            Dataset split. Wikipedia only has ``"train"``.
        limit:
            Stop after this many articles. ``None`` = stream the whole dump
            (~6 M articles for the English Wikipedia).

        Yields
        ------
        CorpusDoc
            One per article, in stream order.
        """
        try:
            from datasets import load_dataset  # noqa: PLC0415  -- optional dep
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "The 'datasets' library is required to stream Wikipedia. "
                "Install it with: uv sync --extra wiki"
            ) from exc

        ds = load_dataset(dataset_id, config, split=split, streaming=True, trust_remote_code=False)
        for count, row in enumerate(ds):
            if limit is not None and count >= limit:
                break
            yield CorpusDoc(
                doc_id=str(row.get("id", count)),
                title=str(row.get("title") or ""),
                text=str(row.get("text") or ""),
            )

    @staticmethod
    def from_fixtures(path: Path) -> Iterator[CorpusDoc]:
        """Read a JSONL fixture file and yield one ``CorpusDoc`` per line.

        Each line must be a JSON object with the keys:
        ``id`` (or ``doc_id``), ``title``, ``text``.

        Lines that are blank or parse as JSON ``null`` are silently skipped.
        """
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"fixtures file {path} line {lineno}: invalid JSON — {exc}"
                    ) from exc
                if obj is None:
                    continue
                doc_id_raw = obj.get("id") or obj.get("doc_id")
                if doc_id_raw is None:
                    raise ValueError(
                        f"fixtures file {path} line {lineno}: missing 'id' field"
                    )
                yield CorpusDoc(
                    doc_id=str(doc_id_raw),
                    title=str(obj.get("title") or ""),
                    text=str(obj.get("text") or ""),
                )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

_PARAGRAPH_SEP = "\n\n"


def _split_into_sub_paras(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Split paragraphs into sub-paragraphs, hard-splitting at chunk_size."""
    sub_paras: list[str] = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) <= chunk_size:
            sub_paras.append(stripped)
        else:
            # Hard split at chunk_size boundaries.
            for start in range(0, len(stripped), chunk_size):
                piece = stripped[start : start + chunk_size]
                if piece:
                    sub_paras.append(piece)
    return sub_paras


def _chunks_for_doc(doc: CorpusDoc, chunk_size: int, overlap: int) -> Iterator[CorpusChunk]:
    """Yield chunks for a single document (inner loop of ``iter_chunks``)."""
    text = doc.text.strip()
    fallback_text = doc.title or text[:chunk_size]

    if not text:
        yield CorpusChunk(
            chunk_id=f"{doc.doc_id}_0",
            doc_id=doc.doc_id,
            title=doc.title,
            text=fallback_text,
        )
        return

    sub_paras = _split_into_sub_paras(text.split(_PARAGRAPH_SEP), chunk_size)

    if not sub_paras:
        yield CorpusChunk(
            chunk_id=f"{doc.doc_id}_0",
            doc_id=doc.doc_id,
            title=doc.title,
            text=fallback_text,
        )
        return

    buffer = ""
    chunk_idx = 0

    for para in sub_paras:
        candidate = (buffer + " " + para).strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        elif buffer:
            yield CorpusChunk(
                chunk_id=f"{doc.doc_id}_{chunk_idx}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=buffer,
            )
            chunk_idx += 1
            # Carry the last `overlap` chars as context for the next chunk.
            # If prefix + para would still exceed chunk_size, start fresh.
            if overlap > 0:
                prefix = buffer[-overlap:].strip()
                new_buf = (prefix + " " + para).strip()
                buffer = new_buf if len(new_buf) <= chunk_size else para
            else:
                buffer = para
        else:
            buffer = para

    if buffer:
        yield CorpusChunk(
            chunk_id=f"{doc.doc_id}_{chunk_idx}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=buffer,
        )


def iter_chunks(
    corpus: Iterator[CorpusDoc],
    chunk_size: int = 512,
    overlap: int = 64,
) -> Iterator[CorpusChunk]:
    """Split long articles into overlapping token-window chunks.

    Algorithm
    ---------
    1. Split the article text on ``\\n\\n`` (paragraph boundaries).
    2. Accumulate paragraphs into a buffer until it would exceed
       ``chunk_size`` characters.
    3. When the buffer hits the limit, emit a ``CorpusChunk`` and slide the
       window forward by ``(chunk_size - overlap)`` characters (keeping the
       last ``overlap`` chars as the start of the next chunk for context).
    4. Paragraphs whose own length exceeds ``chunk_size`` are hard-split at
       ``chunk_size`` boundaries (character-level) before accumulation.

    Parameters
    ----------
    corpus:
        Any iterable of ``CorpusDoc`` objects.
    chunk_size:
        Maximum chunk length in characters. Articles shorter than this
        produce exactly one chunk.
    overlap:
        Number of characters to carry over from the end of one chunk into
        the start of the next (for sliding-window context). Must be less
        than ``chunk_size``.

    Yields
    ------
    CorpusChunk
        In document order; chunk index resets for each new article.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap!r}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    for doc in corpus:
        yield from _chunks_for_doc(doc, chunk_size, overlap)


__all__ = [
    "CorpusChunk",
    "CorpusDoc",
    "WikipediaCorpus",
    "iter_chunks",
]
