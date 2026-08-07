"""Tests for the chunker."""

from __future__ import annotations

from auto_rag.ingestion.chunking import Chunker
from auto_rag.ingestion.loaders import PageContent


def test_chunks_stay_within_size_limit() -> None:
    chunker = Chunker(size=50, overlap=10)
    text = ("This is a fairly long sentence about automotive repairs. " * 10)
    chunks = chunker.chunk_text(text)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_overlap_applied_between_chunks() -> None:
    chunker = Chunker(size=200, overlap=40)
    text = "Sentence one here. " * 20
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    overlap_text = chunks[1].split(". ", 1)[0] + "."
    assert overlap_text in chunks[0]


def test_single_sentence_longer_than_size_is_hard_split() -> None:
    chunker = Chunker(size=30, overlap=5)
    text = "X" * 100
    chunks = chunker.chunk_text(text)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0] == "X" * 30
    assert len(chunks) == 4


def test_invalid_overlap_rejected() -> None:
    try:
        Chunker(size=100, overlap=150)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for overlap >= size")


def test_chunk_pages_attach_metadata() -> None:
    chunker = Chunker(size=100, overlap=10)
    pages = [
        PageContent(page_number=1, text="The camshaft position sensor is located at the front."),
        PageContent(page_number=2, text="Torque the cover bolts to 15 Nm."),
    ]
    chunks = chunker.chunk_pages(pages, {"title": "Manual", "doc_type": "service_manual"})
    assert len(chunks) == 2
    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert chunks[0].metadata["title"] == "Manual"
    assert chunks[1].metadata["doc_type"] == "service_manual"


def test_empty_pages_are_skipped() -> None:
    chunker = Chunker(size=100, overlap=10)
    chunks = chunker.chunk_pages([PageContent(page_number=1, text="   ")], {})
    assert chunks == []
