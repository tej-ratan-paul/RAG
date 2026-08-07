"""LangGraph chat workflow for grounded RAG answers.

The workflow is intentionally a two-node graph so the generation step can be
replaced or surrounded later (e.g. a follow-up retrieval node or a guardrail
node) without touching the service layer:

    START -> retrieve -> generate -> END

``retrieve`` runs the Phase 4 pipeline and formats numbered citations/context;
``generate`` renders the prompt (system + history + question + context) and
calls the Phase 5 LLM. The same node logic is reused by the streaming path.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from auto_rag.llm.base import LLM
from auto_rag.llm.models import ChatMessage
from auto_rag.rag.citations import build_citations, format_context
from auto_rag.rag.models import Citation
from auto_rag.rag.prompts import (
    build_prompt_template,
    from_langchain_messages,
    to_langchain_messages,
)
from auto_rag.retrieval.models import RetrievalFilter, RetrievedChunk
from auto_rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

__all__ = [
    "RAGState",
    "RAGGraph",
    "compute_confidence",
    "build_safety_notes",
]


class RAGState(TypedDict, total=False):
    """State flowing through the LangGraph workflow."""

    question: str
    retrieval_filter: dict[str, Any]
    top_k: int | None
    history: list[ChatMessage]
    chunks: list[RetrievedChunk]
    context: str
    citations: list[Citation]
    answer: str
    confidence: float | None
    safety_notes: list[str]
    model: str


def compute_confidence(chunks: list[RetrievedChunk]) -> float | None:
    """Best available retrieval score as a rough confidence proxy.

    Returns ``None`` when no chunks were retrieved so callers can distinguish
    "no evidence" from "low evidence".
    """
    if not chunks:
        return None
    return round(max(float(chunk.score) for chunk in chunks), 2)


def build_safety_notes(chunks: list[RetrievedChunk]) -> list[str]:
    """Standard safety disclaimers, plus a warning when nothing was found."""
    notes = [
        "Advice is informational only; always follow the latest manufacturer "
        "service information.",
        "Verify torque specifications and fluid types before performing repairs.",
    ]
    if not chunks:
        notes.insert(
            0,
            "No matching documentation was found; do not proceed with a repair "
            "without manufacturer guidance.",
        )
    return notes


class RAGGraph:
    """Compiled LangGraph workflow combining retrieval and generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        prompt=None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt or build_prompt_template()
        self._compiled = self._compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def retrieve_node(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"]
        retrieval_filter = RetrievalFilter(**(state.get("retrieval_filter") or {}))
        top_k = state.get("top_k")
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            retrieval_filter=retrieval_filter,
        )
        logger.debug(
            "Retrieved %d chunks for %r", len(chunks), question[:80]
        )
        return {
            "chunks": chunks,
            "citations": build_citations(chunks),
            "context": format_context(chunks),
        }

    def generate_node(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = self.build_messages(state)
        response = self.llm.generate(messages)
        chunks = state.get("chunks") or []
        return {
            "answer": response.content,
            "confidence": compute_confidence(chunks),
            "safety_notes": build_safety_notes(chunks),
            "model": response.model,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def build_messages(self, state: dict[str, Any]) -> list[ChatMessage]:
        """Render system + history + question/context into LLM messages."""
        template = self.prompt.invoke(
            {
                "question": state["question"],
                "context": state.get("context") or "",
                "history": to_langchain_messages(state.get("history") or []),
            }
        )
        return from_langchain_messages(template.messages)

    def _initial_state(
        self,
        question: str,
        retrieval_filter: RetrievalFilter | None,
        top_k: int | None,
        history: list[ChatMessage] | None,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "retrieval_filter": (
                retrieval_filter.as_dict() if retrieval_filter else {}
            ),
            "top_k": top_k,
            "history": list(history or []),
        }

    def _compile(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate", self.generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def build(self):
        """Return the compiled :class:`StateGraph` instance."""
        return self._compiled

    def invoke(
        self,
        question: str,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int | None = None,
        history: list[ChatMessage] | None = None,
    ) -> RAGState:
        """Run the full retrieve -> generate workflow and return final state."""
        state = self._initial_state(question, retrieval_filter, top_k, history)
        return self._compiled.invoke(state)

    def retrieve_state(
        self,
        question: str,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int | None = None,
        history: list[ChatMessage] | None = None,
    ) -> dict[str, Any]:
        """Run only the retrieve node (used by the streaming path)."""
        state = self._initial_state(question, retrieval_filter, top_k, history)
        return {**state, **self.retrieve_node(state)}
