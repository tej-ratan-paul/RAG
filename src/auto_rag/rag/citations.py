"""Citation and context formatting for the RAG pipeline.

Retrieved chunks are numbered ``[1]``, ``[2]``, ... so the LLM can reference
sources inline; the same numbering is preserved in the returned
:class:`~auto_rag.rag.models.Citation` list so answers stay traceable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from auto_rag.rag.models import Citation
from auto_rag.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

__all__ = ["build_citations", "format_context", "format_citations"]

_SNIPPET_MAX_CHARS: int = 300
_SNIPPET_ELLIPSIS: str = "…"


def _chunk_source(chunk: RetrievedChunk) -> str:
    """Chunk source, falling back to metadata when not on the model."""
    return chunk.source or chunk.metadata.get("source", "")


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map ranked chunks onto numbered, deduplicated citations."""
    citations: list[Citation] = []
    seen_sources: set[tuple[str, int | None]] = set()
    for chunk in chunks:
        source = _chunk_source(chunk)
        key = (source, chunk.page)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        citations.append(
            Citation(
                index=len(citations) + 1,
                source=source,
                score=round(float(chunk.score), 4),
                page=chunk.page,
                doc_type=chunk.doc_type,
                make=chunk.make,
                model=chunk.model,
                snippet=_truncate(chunk.text),
            )
        )
    logger.debug("Built %d citations from %d chunks", len(citations), len(chunks))
    return citations


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered passages with source/page headers.

    The ``[n]`` markers are the same numbers used in :func:`build_citations`,
    so model output such as ``see [2]`` maps straight back to a citation.
    """
    if not chunks:
        return ""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = Path(_chunk_source(chunk)).name if _chunk_source(chunk) else "unknown"
        page = f", page {chunk.page}" if chunk.page is not None else ""
        parts.append(f"[{index}] ({source}{page}) {chunk.text}")
    return "\n\n".join(parts)


def format_citations(citations: list[Citation]) -> str:
    """Render citations as a human-readable source list (e.g. for the CLI)."""
    if not citations:
        return ""
    lines: list[str] = []
    for citation in citations:
        source = Path(citation.source).name if citation.source else "unknown"
        page = f"p{citation.page}" if citation.page is not None else ""
        vehicle = " ".join(
            part for part in (citation.make, citation.model) if part
        )
        location = ", ".join(part for part in (page, vehicle, citation.doc_type) if part)
        suffix = f" [{location}]" if location else ""
        lines.append(f"[{citation.index}] {source}{suffix} (score {citation.score:.3f})")
    return "\n".join(lines)


def _truncate(text: str) -> str:
    """Trim long chunk text to a bounded snippet for citation display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= _SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[:_SNIPPET_MAX_CHARS].rstrip() + _SNIPPET_ELLIPSIS
