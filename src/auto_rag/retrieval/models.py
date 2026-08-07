"""Retrieval data models.

Shared structures for the retrieval pipeline: a uniform chunk result and a
filter that narrows retrieval by vehicle / document metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

from auto_rag.constants import DOCUMENT_TYPES

__all__ = ["RetrievedChunk", "RetrievalFilter", "FILTER_FIELDS"]

FILTER_FIELDS: Final[tuple[str, ...]] = ("make", "model", "year", "doc_type", "vin")


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval result with its provenance and score."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    source: str = ""

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> RetrievedChunk:
        """Build from a vector-store hit dict (``id/text/metadata/score``)."""
        metadata = hit.get("metadata") or {}
        return cls(
            id=hit["id"],
            text=hit["text"],
            metadata=metadata,
            score=float(hit.get("score") or 0.0),
            source=metadata.get("source", ""),
        )

    def with_score(self, score: float) -> RetrievedChunk:
        """Return a copy carrying a new score."""
        return replace(self, score=score)

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page is not None else None

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "")

    @property
    def make(self) -> str:
        return self.metadata.get("make", "")

    @property
    def model(self) -> str:
        return self.metadata.get("model", "")


@dataclass(frozen=True)
class RetrievalFilter:
    """Optional narrowing criteria applied across both retrieval channels.

    Only the fields set to a non-None value participate in filtering; empty
    strings stored in metadata are treated the same as ``None``.
    """

    make: str | None = None
    model: str | None = None
    year: int | None = None
    doc_type: str | None = None
    vin: str | None = None

    @classmethod
    def from_doc_type(cls, doc_type: str) -> RetrievalFilter:
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type {doc_type!r}")
        return cls(doc_type=doc_type)

    @property
    def active(self) -> bool:
        """True when at least one criterion is set."""
        return any(getattr(self, field) is not None for field in FILTER_FIELDS)

    def as_dict(self) -> dict[str, Any]:
        """Return only the active criteria as ``{field: value}``."""
        return {
            field: getattr(self, field)
            for field in FILTER_FIELDS
            if getattr(self, field) is not None
        }
