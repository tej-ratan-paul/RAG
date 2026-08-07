"""Tests for citation and context formatting."""

from __future__ import annotations

from auto_rag.rag.citations import build_citations, format_citations, format_context
from auto_rag.retrieval.models import RetrievedChunk


def _chunk(text: str, *, source: str = "manual.pdf", page: int = 3, score: float = 0.9, **meta) -> RetrievedChunk:
    metadata = {
        "source": source,
        "page": page,
        "doc_type": "service_manual",
        "make": "Toyota",
        "model": "Camry",
        **meta,
    }
    return RetrievedChunk(id=f"{source}:{page}", text=text, metadata=metadata, score=score)


def test_build_citations_assigns_consecutive_indexes() -> None:
    chunks = [
        _chunk("first", page=1),
        _chunk("second", page=2),
        _chunk("third", page=3),
    ]
    citations = build_citations(chunks)
    assert [c.index for c in citations] == [1, 2, 3]
    assert citations[0].source == "manual.pdf"
    assert citations[0].page == 1
    assert citations[0].score == 0.9
    assert citations[0].make == "Toyota"


def test_build_citations_deduplicates_same_source_page() -> None:
    chunks = [_chunk("a", page=1), _chunk("b", page=1)]
    citations = build_citations(chunks)
    assert len(citations) == 1
    assert citations[0].index == 1


def test_build_citations_truncates_long_snippet() -> None:
    chunks = [_chunk("word " * 200, page=1)]
    citation = build_citations(chunks)[0]
    assert len(citation.snippet) < len("word " * 200)
    assert citation.snippet.endswith("…")


def test_format_context_numbers_passages() -> None:
    chunks = [_chunk("Change the oil", page=2), _chunk("Replace the filter", page=5)]
    context = format_context(chunks)
    assert context.startswith("[1] (manual.pdf, page 2)")
    assert "[2] (manual.pdf, page 5)" in context
    assert "Change the oil" in context


def test_format_context_empty_returns_empty_string() -> None:
    assert format_context([]) == ""


def test_format_context_omits_page_when_missing() -> None:
    chunks = [_chunk("Just text", page=None)]
    context = format_context(chunks)
    assert "page" not in context


def test_format_citations_renders_source_list() -> None:
    chunks = [_chunk("a", page=2, score=0.812), _chunk("b", page=7, score=0.5)]
    rendered = format_citations(build_citations(chunks))
    assert "[1] manual.pdf" in rendered
    assert "[2] manual.pdf" in rendered
    assert "p2" in rendered
    assert "0.812" in rendered


def test_format_citations_empty_returns_empty_string() -> None:
    assert format_citations([]) == ""
