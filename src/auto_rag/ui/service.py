"""Cached service bootstrap for the Streamlit UI.

Mirrors the wiring in :mod:`auto_rag.rag.cli` but keeps the expensive objects
(alembic-free DB, vector store, BM25 index, reranker, LLM) alive across Streamlit
re-runs via ``st.cache_resource``. The bundle is keyed on the LLM override
signature, so applying new configuration in the UI transparently builds a fresh
bundle without touching other sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from auto_rag.config import get_settings
from auto_rag.db.connection import Database
from auto_rag.db.repositories import ConversationRepository
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.llm import build_llm
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.service import RAGService
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.reranker import build_reranker
from auto_rag.retrieval.retriever import Retriever

__all__ = ["UIServiceBundle", "build_bundle"]


@dataclass
class UIServiceBundle:
    """Everything the UI needs to answer questions and manage sessions."""

    service: RAGService
    conversations: ConversationRepository
    db: Database | None = None


@st.cache_resource(show_spinner=False)
def build_bundle(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> UIServiceBundle:
    """Construct (and cache) the runtime bundle from settings + overrides."""
    settings = get_settings().model_copy(deep=True)
    if provider:
        settings.llm.provider = provider
    if model:
        settings.llm.model = model
    if base_url:
        settings.llm.base_url = base_url
    if temperature is not None:
        settings.llm.temperature = float(temperature)
    settings.prepare_directories()

    db = Database.from_settings(settings)
    db.initialize()
    try:
        store = build_vector_store(settings)
        bm25 = BM25Index(store.get_all_chunks(limit=100_000))
        reranker = build_reranker(
            enabled=settings.retrieval.rerank,
            model_name=settings.retrieval.reranker_model,
            device=settings.embeddings.device,
        )
        retriever = Retriever(
            vector_store=store,
            embedding_provider=store.provider,
            config=settings.retrieval,
            bm25_index=bm25,
            reranker=reranker,
        )
        llm = build_llm(settings.llm)
        repo = ConversationRepository(db)
        service = RAGService(retriever, llm, ConversationMemory(repo))
        return UIServiceBundle(service=service, conversations=repo, db=db)
    except Exception:
        db.close()
        raise
