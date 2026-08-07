"""Structured response models for the RAG service.

:class:`Citation` captures the provenance of a single source passage used to
answer a question. :class:`RAGResult` is the uniform contract returned by
:class:`auto_rag.rag.service.RAGService` for every answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Citation", "RAGResult"]


@dataclass(frozen=True)
class Citation:
    """A numbered reference to a retrieved source passage."""

    index: int
    source: str
    score: float
    page: int | None = None
    doc_type: str = ""
    make: str = ""
    model: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class RAGResult:
    """The complete structured answer for a single user question."""

    query: str
    answer: str
    sources: list[Citation] = field(default_factory=list)
    confidence: float | None = None
    safety_notes: list[str] = field(default_factory=list)
    conversation_id: int | None = None
    model: str = ""

    def has_sources(self) -> bool:
        """True when at least one source passage backed the answer."""
        return bool(self.sources)
