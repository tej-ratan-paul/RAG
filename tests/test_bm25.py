"""Tests for the BM25 lexical index."""

from __future__ import annotations

from auto_rag.retrieval.bm25 import BM25Index, build_bm25_index, tokenize
from auto_rag.retrieval.models import RetrievalFilter


def _chunks() -> list[dict]:
    return [
        {
            "id": "a:0",
            "text": "Front brake pads minimum thickness 2.0 mm.",
            "metadata": {"make": "Toyota", "model": "Camry", "source": "a.pdf"},
        },
        {
            "id": "b:0",
            "text": "Replace the spark plugs at 60000 miles.",
            "metadata": {"make": "Honda", "model": "Civic", "source": "b.pdf"},
        },
        {
            "id": "c:0",
            "text": "Brake fluid flush every two years.",
            "metadata": {"make": "Toyota", "model": "Corolla", "source": "c.pdf"},
        },
    ]


def test_tokenize_keeps_codes() -> None:
    assert tokenize("Check P0300 and replace 02 sensor") == [
        "check",
        "p0300",
        "and",
        "replace",
        "02",
        "sensor",
    ]


def test_tokenize_empty() -> None:
    assert tokenize("  ") == []


def test_bm25_ranks_exact_overlap_first() -> None:
    index = BM25Index(_chunks())
    hits = index.search("brake pads", top_k=3)
    assert hits
    assert hits[0].id == "a:0"
    assert hits[0].score > 0


def test_bm25_prefers_more_term_overlap() -> None:
    index = BM25Index(_chunks())
    hits = index.search("brake fluid", top_k=3)
    assert hits[0].id == "c:0"
    assert "b:0" not in {hit.id for hit in hits}  # no shared tokens at all


def test_bm25_filter_narrows_results() -> None:
    index = BM25Index(_chunks())
    hits = index.search("brake", top_k=3, retrieval_filter=RetrievalFilter(make="Toyota"))
    assert all(hit.metadata["make"] == "Toyota" for hit in hits)
    assert len(hits) >= 1


def test_bm25_empty_query() -> None:
    index = BM25Index(_chunks())
    assert index.search("   ", top_k=3) == []


def test_bm25_len_and_builder() -> None:
    chunks = _chunks()
    index = build_bm25_index(chunks)
    assert len(index) == 3
    assert index.chunks == chunks
