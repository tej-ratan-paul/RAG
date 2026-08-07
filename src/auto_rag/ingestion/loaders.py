"""Document loaders.

Extracts raw text from source files into a list of :class:`PageContent`
objects, preserving page boundaries and per-page raw metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from auto_rag.errors import IngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageContent:
    """Raw text and metadata extracted from a single page."""

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Protocol implemented by all loaders."""

    def load(self, path: Path) -> list[PageContent]:
        """Return one :class:`PageContent` per page in the document."""
        ...


class PDFLoader:
    """Loads text from PDF files using pypdf."""

    def load(self, path: Path) -> list[PageContent]:
        """Extract per-page text from ``path``."""
        if not path.is_file():
            raise IngestionError(f"PDF not found: {path}")
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF {path}: {exc}") from exc

        if reader.is_encrypted:
            raise IngestionError(
                f"Encrypted PDF not supported: {path} (please decrypt before ingestion)"
            )

        pages: list[PageContent] = []
        for index, page in enumerate(reader.pages, start=1):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - pypdf internals vary
                logger.warning("Page %d of %s could not be extracted: %s", index, path, exc)
            pages.append(
                PageContent(page_number=index, text=text, metadata={"page_number": index})
            )
        logger.debug("Loaded %d pages from %s", len(pages), path)
        return pages


class TextLoader:
    """Loads plain-text files (``.txt``, ``.md``) as a single page."""

    def load(self, path: Path) -> list[PageContent]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestionError(f"Failed to read text file {path}: {exc}") from exc
        return [PageContent(page_number=1, text=text, metadata={"page_number": 1})]


SUPPORTED_SUFFIXES: dict[str, DocumentLoader] = {
    ".pdf": PDFLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
}


def loader_for(path: Path) -> DocumentLoader:
    """Return the appropriate loader for ``path``'s suffix."""
    loader = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if loader is None:
        raise IngestionError(
            f"Unsupported file type {path.suffix!r} for {path.name} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    return loader
