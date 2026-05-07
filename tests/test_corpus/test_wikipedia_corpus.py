"""Tests for the Wikipedia corpus loader and chunker.

All tests are offline — no HuggingFace download occurs. The `from_huggingface`
path is tested by patching `datasets.load_dataset` with a fake iterable.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vdbbench.cli import app
from vdbbench.corpus.wikipedia import (
    CorpusDoc,
    WikipediaCorpus,
    iter_chunks,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FIXTURES_PATH = Path(__file__).parent.parent.parent / "evals" / "wikipedia_fixtures.jsonl"


def _make_fake_hf_row(idx: int) -> dict[str, Any]:
    return {
        "id": str(idx),
        "title": f"Article {idx}",
        "text": f"Paragraph one for article {idx}.\n\nParagraph two for article {idx}.",
    }


# ---------------------------------------------------------------------------
# CorpusDoc model
# ---------------------------------------------------------------------------


class TestCorpusDoc:
    def test_frozen(self) -> None:
        doc = CorpusDoc(doc_id="1", title="T", text="body")
        with pytest.raises((AttributeError, TypeError)):
            doc.doc_id = "2"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = CorpusDoc(doc_id="x", title="T", text="body")
        b = CorpusDoc(doc_id="x", title="T", text="body")
        assert a == b


# ---------------------------------------------------------------------------
# from_fixtures
# ---------------------------------------------------------------------------


class TestFromFixtures:
    def test_reads_fixture_file_correctly(self) -> None:
        docs = list(WikipediaCorpus.from_fixtures(FIXTURES_PATH))
        assert len(docs) >= 5, "Expected at least 5 fixture rows"
        first = docs[0]
        assert isinstance(first, CorpusDoc)
        assert first.doc_id  # non-empty id
        assert first.title  # non-empty title
        assert first.text  # non-empty text

    def test_doc_fields_match_jsonl(self) -> None:
        docs = list(WikipediaCorpus.from_fixtures(FIXTURES_PATH))
        # Load raw for cross-check
        raw_rows = [
            json.loads(line)
            for line in FIXTURES_PATH.read_text().splitlines()
            if line.strip()
        ]
        assert len(docs) == len(raw_rows)
        for doc, raw in zip(docs, raw_rows, strict=True):
            assert doc.doc_id == str(raw["id"])
            assert doc.title == raw["title"]
            assert doc.text == raw["text"]

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"title": "No id", "text": "body"}\n')
        with pytest.raises(ValueError, match="missing 'id'"):
            list(WikipediaCorpus.from_fixtures(bad))

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "blank.jsonl"
        f.write_text('\n{"id": "1", "title": "T", "text": "body"}\n\n')
        docs = list(WikipediaCorpus.from_fixtures(f))
        assert len(docs) == 1

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "invalid.jsonl"
        bad.write_text("{not valid json}\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            list(WikipediaCorpus.from_fixtures(bad))


# ---------------------------------------------------------------------------
# from_huggingface (mocked)
# ---------------------------------------------------------------------------


class TestFromHuggingface:
    def test_yields_corpus_docs(self) -> None:
        fake_rows = [_make_fake_hf_row(i) for i in range(3)]
        with patch("vdbbench.corpus.wikipedia.WikipediaCorpus.from_huggingface") as mock_hf:
            mock_hf.return_value = iter(
                [
                    CorpusDoc(
                        doc_id=str(r["id"]),
                        title=r["title"],
                        text=r["text"],
                    )
                    for r in fake_rows
                ]
            )
            docs = list(WikipediaCorpus.from_huggingface(limit=3))
        assert len(docs) == 3
        assert all(isinstance(d, CorpusDoc) for d in docs)

    def test_from_huggingface_respects_limit_via_datasets_mock(self) -> None:
        """Patch `datasets.load_dataset` and verify limit truncates output."""
        fake_rows = [_make_fake_hf_row(i) for i in range(20)]
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_rows))

        with patch("datasets.load_dataset", return_value=mock_ds):
            docs = list(WikipediaCorpus.from_huggingface(limit=5))

        assert len(docs) == 5
        for i, doc in enumerate(docs):
            assert isinstance(doc, CorpusDoc)
            assert doc.doc_id == str(i)
            assert doc.title == f"Article {i}"

    def test_from_huggingface_shape_is_correct(self) -> None:
        fake_rows = [_make_fake_hf_row(i) for i in range(4)]
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_rows))

        with patch("datasets.load_dataset", return_value=mock_ds):
            docs = list(WikipediaCorpus.from_huggingface(limit=None))

        assert len(docs) == 4
        assert docs[0].title == "Article 0"
        assert "Paragraph one" in docs[0].text

    def test_missing_datasets_raises_import_error(self) -> None:
        # Patch the `load_dataset` name inside the wikipedia module's namespace
        # so it looks like the import failed even when datasets is installed.
        import builtins  # noqa: PLC0415

        import vdbbench.corpus.wikipedia as wiki_mod  # noqa: PLC0415

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "datasets":
                raise ImportError("datasets")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        importlib.reload(wiki_mod)
        with patch("builtins.__import__", side_effect=_fake_import), pytest.raises(
            ImportError, match="datasets"
        ):
            list(wiki_mod.WikipediaCorpus.from_huggingface(limit=1))


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


class TestIterChunks:
    def test_short_article_produces_one_chunk(self) -> None:
        doc = CorpusDoc(doc_id="1", title="Short", text="This is a short article.")
        chunks = list(iter_chunks(iter([doc]), chunk_size=512, overlap=64))
        assert len(chunks) == 1
        assert chunks[0].doc_id == "1"
        assert chunks[0].title == "Short"

    def test_long_article_produces_multiple_chunks(self) -> None:
        # Two long paragraphs that together exceed chunk_size=100.
        para1 = "A" * 80
        para2 = "B" * 80
        text = para1 + "\n\n" + para2
        doc = CorpusDoc(doc_id="2", title="Long", text=text)
        chunks = list(iter_chunks(iter([doc]), chunk_size=100, overlap=10))
        assert len(chunks) >= 2

    def test_chunk_ids_use_doc_id_prefix(self) -> None:
        doc = CorpusDoc(doc_id="myid", title="T", text="Word " * 300)
        chunks = list(iter_chunks(iter([doc]), chunk_size=100, overlap=10))
        assert all(c.chunk_id.startswith("myid_") for c in chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        doc = CorpusDoc(doc_id="x", title="T", text=("Para\n\n" * 20))
        chunks = list(iter_chunks(iter([doc]), chunk_size=50, overlap=5))
        indices = [int(c.chunk_id.split("_")[-1]) for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_text_does_not_exceed_chunk_size(self) -> None:
        text = "\n\n".join(["Word " * 50] * 10)
        doc = CorpusDoc(doc_id="3", title="T", text=text)
        chunks = list(iter_chunks(iter([doc]), chunk_size=200, overlap=20))
        for chunk in chunks:
            assert len(chunk.text) <= 200, f"Chunk too long: {len(chunk.text)}"

    def test_overlap_carries_context(self) -> None:
        # Create two paragraphs that force a boundary.
        para1 = "Alpha " * 30  # 180 chars
        para2 = "Beta " * 30  # 180 chars
        text = para1.strip() + "\n\n" + para2.strip()
        doc = CorpusDoc(doc_id="4", title="T", text=text)
        chunks = list(iter_chunks(iter([doc]), chunk_size=200, overlap=50))
        # First chunk must contain some Alpha content.
        assert "Alpha" in chunks[0].text

    def test_empty_text_yields_title_chunk(self) -> None:
        doc = CorpusDoc(doc_id="5", title="MyTitle", text="")
        chunks = list(iter_chunks(iter([doc]), chunk_size=512, overlap=64))
        assert len(chunks) == 1
        assert chunks[0].text == "MyTitle"

    def test_multiple_docs_independent_chunk_indices(self) -> None:
        docs = [
            CorpusDoc(doc_id="d1", title="T1", text=("Para\n\n" * 15)),
            CorpusDoc(doc_id="d2", title="T2", text=("Para\n\n" * 15)),
        ]
        chunks = list(iter_chunks(iter(docs), chunk_size=50, overlap=5))
        d1_chunks = [c for c in chunks if c.doc_id == "d1"]
        d2_chunks = [c for c in chunks if c.doc_id == "d2"]
        # Both start from index 0.
        assert d1_chunks[0].chunk_id == "d1_0"
        assert d2_chunks[0].chunk_id == "d2_0"

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            list(iter_chunks(iter([]), chunk_size=0))

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            list(iter_chunks(iter([]), chunk_size=100, overlap=100))

    def test_paragraph_boundary_respected(self) -> None:
        # Para1 fits inside chunk; para2 forces a new chunk.
        para1 = "Short paragraph."
        para2 = "X" * 90
        text = para1 + "\n\n" + para2
        doc = CorpusDoc(doc_id="6", title="T", text=text)
        chunks = list(iter_chunks(iter([doc]), chunk_size=100, overlap=0))
        # Both paragraphs must appear in some chunk.
        all_text = " ".join(c.text for c in chunks)
        assert "Short paragraph" in all_text
        assert "X" * 10 in all_text


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliWikipedia:
    def test_prep_wikipedia_without_limit_errors(self) -> None:
        """vdbbench prep --dataset wikipedia without --limit or --fixtures -> exit 2."""
        result = CliRunner().invoke(app, ["prep", "--dataset", "wikipedia"])
        assert result.exit_code == 2
        # Should mention limit or wikipedia in the output.
        assert "wikipedia" in result.output.lower() or "limit" in result.output.lower()

    def test_prep_wikipedia_help_shows_options(self) -> None:
        """vdbbench prep --help mentions wikipedia and the new flags."""
        result = CliRunner().invoke(app, ["prep", "--help"])
        assert result.exit_code == 0
        assert "wikipedia" in result.output.lower()
        assert "--limit" in result.output
        assert "--fixtures" in result.output

    def test_prep_wikipedia_fixture_end_to_end(self, tmp_path: Path) -> None:
        """Full pipeline: fixtures -> iter_chunks -> FakeEncoder -> encoded dir."""
        import pandas as pd  # noqa: PLC0415

        from vdbbench.corpus.bundle import CorpusBundle  # noqa: PLC0415
        from vdbbench.corpus.wikipedia import WikipediaCorpus  # noqa: PLC0415
        from vdbbench.corpus.wikipedia import iter_chunks as wiki_iter_chunks  # noqa: PLC0415
        from vdbbench.embed import FakeEncoder, encode_corpus  # noqa: PLC0415

        docs = list(WikipediaCorpus.from_fixtures(FIXTURES_PATH))
        chunks = list(wiki_iter_chunks(iter(docs), chunk_size=200, overlap=20))
        assert chunks

        passage_rows = [
            {"pid": c.chunk_id, "text": (c.title + " " + c.text).strip()} for c in chunks
        ]
        passages = pd.DataFrame(passage_rows)
        query_rows = [{"qid": "wq_0", "text": chunks[0].title or chunks[0].text[:80]}]
        queries = pd.DataFrame(query_rows)
        qrel_rows = [{"qid": "wq_0", "pid": chunks[0].chunk_id, "relevance": 1.0}]
        qrels = pd.DataFrame(qrel_rows)

        bundle = CorpusBundle(
            name="wikipedia-test",
            passages=passages,
            queries=queries,
            qrels=qrels,
            metadata={"dataset": "wikipedia"},
        )
        enc = FakeEncoder(dim=32)
        encoded = encode_corpus(bundle, enc)
        out = tmp_path / "encoded"
        encoded.save(out)

        assert (out / "manifest.json").is_file()
        assert (out / "passages.parquet").is_file()
        assert (out / "queries.parquet").is_file()
        assert (out / "corpus").is_dir()
