"""RAG service: the public facade over the chat workflow.

Binds the Phase 3/4 retrieval pipeline, the Phase 5 LLM layer, and durable
conversation memory together. ``ask`` returns a complete :class:`RAGResult`;
``ask_stream`` yields answer text incrementally and exposes the final
:class:`RAGResult` via :attr:`last_result`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from auto_rag.llm.base import LLM
from auto_rag.rag.graph import (
    RAGGraph,
    build_safety_notes,
    compute_confidence,
)
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.models import RAGResult
from auto_rag.rag.prompts import build_prompt_template
from auto_rag.retrieval.models import RetrievalFilter
from auto_rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

__all__ = ["RAGService"]


class RAGService:
    """Answer repair questions with grounded, cited responses."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        memory: ConversationMemory,
        *,
        prompt=None,
        default_top_k: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.memory = memory
        self.default_top_k = default_top_k
        self.graph = RAGGraph(retriever, llm, prompt or build_prompt_template())
        self.last_result: RAGResult | None = None

    def ask(
        self,
        question: str,
        *,
        conversation_id: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int | None = None,
        title: str | None = None,
    ) -> RAGResult:
        """Answer ``question`` and persist the exchange in conversation memory."""
        conversation = self.memory.get_or_create(conversation_id, title=title)
        history = self.memory.to_llm_history(conversation.id)
        state = self.graph.invoke(
            question,
            retrieval_filter=retrieval_filter,
            top_k=top_k or self.default_top_k,
            history=history,
        )
        result = self._to_result(
            question=question,
            conversation_id=conversation.id,
            state=state,
        )
        self._persist(conversation.id, question, result)
        self.last_result = result
        return result

    def ask_stream(
        self,
        question: str,
        *,
        conversation_id: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int | None = None,
    ) -> Iterator[str]:
        """Stream the answer token by token, then persist the exchange.

        After the iterator is exhausted, the final :class:`RAGResult` is
        available on :attr:`last_result`. Iterate fully to ensure persistence.
        """
        conversation = self.memory.get_or_create(conversation_id)
        history = self.memory.to_llm_history(conversation.id)
        prepared = self.graph.retrieve_state(
            question,
            retrieval_filter=retrieval_filter,
            top_k=top_k or self.default_top_k,
            history=history,
        )
        messages = self.graph.build_messages(prepared)

        parts: list[str] = []
        for piece in self.llm.generate_stream(messages):
            parts.append(piece)
            yield piece

        answer = "".join(parts)
        chunks = prepared.get("chunks") or []
        result = RAGResult(
            query=question,
            answer=answer,
            sources=prepared.get("citations") or [],
            confidence=compute_confidence(chunks),
            safety_notes=build_safety_notes(chunks),
            conversation_id=conversation.id,
            model=self.llm.model_name,
        )
        self._persist(conversation.id, question, result)
        self.last_result = result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _to_result(
        self,
        *,
        question: str,
        conversation_id: int,
        state: dict[str, Any],
    ) -> RAGResult:
        return RAGResult(
            query=question,
            answer=state.get("answer") or "",
            sources=state.get("citations") or [],
            confidence=state.get("confidence"),
            safety_notes=state.get("safety_notes") or [],
            conversation_id=conversation_id,
            model=state.get("model") or "",
        )

    def _persist(
        self,
        conversation_id: int,
        question: str,
        result: RAGResult,
    ) -> None:
        self.memory.add_user_message(conversation_id, question)
        self.memory.add_assistant_message(
            conversation_id,
            result.answer,
            citations=result.sources or None,
        )
        logger.debug(
            "Persisted turn for conversation %s (%d sources)",
            conversation_id,
            len(result.sources),
        )
