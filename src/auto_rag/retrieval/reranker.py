"""Cross-encoder reranking.

A cross-encoder scores query+chunk pairs jointly, giving a stronger relevance
signal than bi-encoder dot products. The encoder is imported lazily so module
imports stay cheap when reranking is disabled. A :class:`NoopReranker` keeps
the pipeline uniform when reranking is turned off.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from auto_rag.ingestion.embeddings import resolve_device
from auto_rag.retrieval.models import RetrievedChunk

__all__ = ["Reranker", "NoopReranker", "CrossEncoderReranker", "build_reranker"]


class Reranker(Protocol):
    """Re-orders a list of chunks by query relevance."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks, best first."""
        ...


class NoopReranker:
    """Identity reranker used when reranking is disabled."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        return list(chunks[:top_k])


class CrossEncoderReranker:
    """Reranks chunks with a cross-encoder model."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        resolved = resolve_device(device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=resolved)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(
            zip(chunks, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            replace(chunk, score=float(score)) for chunk, score in ranked[:top_k]
        ]


def build_reranker(
    *,
    enabled: bool,
    model_name: str,
    device: str = "auto",
) -> Reranker:
    """Return a reranker based on ``enabled`` (no-op when disabled)."""
    if not enabled:
        return NoopReranker()
    return CrossEncoderReranker(model_name=model_name, device=device)
