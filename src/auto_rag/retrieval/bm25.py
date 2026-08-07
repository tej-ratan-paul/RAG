"""BM25 lexical index.

Provides a keyword-overlap search channel that complements dense embeddings:
rare part numbers, DTC codes and exact phrases that embeddings may blur are
matched precisely here. The index is built from the chunks already stored in
the vector store and lives entirely in memory.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from auto_rag.retrieval.filters import matches_filter
from auto_rag.retrieval.models import RetrievalFilter, RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_TOKENIZER_CACHE: dict[str, list[str]] = {}


def tokenize(text: str) -> list[str]:
    """Split text into lowercased alphanumeric tokens.

    ``P0300`` survives as a single token, which matters for DTC lookups.
    """
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


class BM25Index:
    """An in-memory BM25Okapi index over a fixed set of chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        """Build an index from vector-store chunk dicts (``id/text/metadata``)."""
        self._chunks = chunks
        self._ids = [chunk["id"] for chunk in chunks]
        self._texts = [chunk["text"] for chunk in chunks]
        self._metadatas = [chunk.get("metadata") or {} for chunk in chunks]
        corpus = [tokenize(text) for text in self._texts]
        self._model = BM25Okapi(corpus)
        logger.info("Built BM25 index over %d chunks", len(chunks))

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Raw chunks the index was built from (read-only view)."""
        return list(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Rank stored chunks by BM25 relevance to ``query``.

        Args:
            query: The user query (tokenised internally).
            top_k: Maximum number of results to return.
            retrieval_filter: Optional metadata narrowing applied after scoring.

        Returns:
            Matches with a BM25 score, most relevant first. Chunks with no
            lexical overlap (score <= 0) are excluded.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._model.get_scores(query_tokens)

        ranked: list[RetrievedChunk] = []
        for index in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            score = scores[index]
            if score <= 0.0:
                break
            metadata = self._metadatas[index]
            if retrieval_filter is not None and not matches_filter(
                retrieval_filter, metadata
            ):
                continue
            ranked.append(
                RetrievedChunk(
                    id=self._ids[index],
                    text=self._texts[index],
                    metadata=metadata,
                    score=float(score),
                    source=metadata.get("source", ""),
                )
            )
            if len(ranked) >= top_k:
                break
        return ranked


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Index:
    """Construct a :class:`BM25Index` from vector-store chunks."""
    return BM25Index(chunks)
