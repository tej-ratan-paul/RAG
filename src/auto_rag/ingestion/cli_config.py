"""Shared wiring helpers for ingestion CLI entry points."""

from __future__ import annotations

from auto_rag.config import Settings
from auto_rag.ingestion.embeddings import EmbeddingProvider, get_embedding_provider
from auto_rag.ingestion.vectorstore import VectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct a :class:`VectorStore` from application settings."""
    provider: EmbeddingProvider = get_embedding_provider(
        settings.embeddings, cache_path=settings.embedding_cache_path
    )
    return VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.vectorstore.collection_name,
        provider=provider,
        distance=settings.vectorstore.distance,
    )
