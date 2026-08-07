"""Document chunking.

Splits cleaned page text into overlapping, sentence-aware chunks suitable for
embedding. Each chunk carries stable metadata (source, page, index) used for
retrieval filtering and citation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import tiktoken

from auto_rag.ingestion.loaders import PageContent

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\"'(])")
_TOKENIZER_NAME = "o200k_base"
_encoding = tiktoken.get_encoding(_TOKENIZER_NAME)


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text`` using tiktoken."""
    return len(_encoding.encode(text))


@dataclass(frozen=True)
class Chunk:
    """A single embeddable chunk with rich metadata."""

    text: str
    metadata: dict[str, Any]

    def with_metadata(self, extra: dict[str, Any]) -> Chunk:
        """Return a copy of this chunk with merged extra metadata."""
        merged = {**self.metadata, **extra}
        return Chunk(text=self.text, metadata=merged)


class Chunker:
    """Sentence-aware overlapping chunker with a configurable size."""

    def __init__(self, size: int, overlap: int) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping chunks, each at most ``size`` chars."""
        sentences = self._split_sentences(text)
        chunks: list[str] = []
        buffer = ""

        for sentence in sentences:
            if not sentence.strip():
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= self.size:
                buffer = candidate
                continue

            # Buffer full: flush and start a new one with overlap.
            if buffer:
                chunks.append(buffer)
                buffer = self._tail(buffer)
            else:
                buffer = ""

            # A single sentence may itself exceed the limit: hard split it.
            remainder = sentence
            while len(remainder) > self.size:
                piece, remainder = remainder[: self.size], remainder[self.size :]
                chunks.append(piece)
                buffer = self._tail(piece)
            if remainder:
                buffer = f"{buffer} {remainder}".strip()

        if buffer:
            chunks.append(buffer)
        return chunks

    def chunk_pages(
        self,
        pages: list[PageContent],
        document: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk loaded pages, attaching document and page metadata to each."""
        chunks: list[Chunk] = []
        index = 0
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            for piece in self.chunk_text(text):
                metadata = {
                    **document,
                    "page": page.page_number,
                    "chunk_index": index,
                }
                chunks.append(Chunk(text=piece, metadata=metadata))
                index += 1
        logger.debug("Produced %d chunks (size=%d, overlap=%d)", index, self.size, self.overlap)
        return chunks

    # ------------------------------------------------------------------ #
    def _split_sentences(self, text: str) -> list[str]:
        parts = _SENTENCE_BOUNDARY.split(text)
        return [part.strip() for part in parts if part.strip()]

    def _tail(self, text: str) -> str:
        """Return the trailing overlap of ``text`` (sentence boundary aware)."""
        if self.overlap <= 0 or len(text) <= self.overlap:
            return ""
        tail = text[-self.overlap :]
        boundary = tail.find(" ")
        if boundary != -1:
            tail = tail[boundary:].lstrip()
        return tail
