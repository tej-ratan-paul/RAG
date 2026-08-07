"""Tests for the LangGraph RAG workflow."""

from __future__ import annotations

from auto_rag.llm.models import ChatMessage, LLMResponse
from auto_rag.rag.graph import RAGGraph, build_safety_notes, compute_confidence
from auto_rag.retrieval.models import RetrievedChunk


class FakeLLM:
    """Deterministic LLM double implementing the LLM protocol."""

    def __init__(self, response: str = "Brake pads need 25 Nm [1].") -> None:
        self.model_name = "fake-model"
        self._response = response
        self.generated: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens
        self.generated.append(messages)
        return LLMResponse(content=self._response, model=self.model_name)

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> object:
        del messages, temperature, max_tokens
        return iter(())


class FakeRetriever:
    """Retrieval double returning a fixed chunk list."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls: list[tuple] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieval_filter=None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, retrieval_filter))
        return list(self.chunks)


def _chunk(text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        id="c1",
        text=text,
        metadata={"source": "manual.pdf", "page": 4, "doc_type": "service_manual"},
        score=score,
    )


def _graph(retriever: FakeRetriever, llm: FakeLLM) -> RAGGraph:
    return RAGGraph(retriever, llm)


def test_invoke_runs_retrieve_then_generate() -> None:
    retriever = FakeRetriever([_chunk("Torque to 25 Nm.")])
    llm = FakeLLM(response="Torque to 25 Nm [1].")
    state = _graph(retriever, llm).invoke("What torque for brake calipers?")

    assert state["answer"] == "Torque to 25 Nm [1]."
    assert state["model"] == "fake-model"
    assert state["confidence"] == 0.8
    assert state["context"].startswith("[1] (manual.pdf, page 4)")
    assert len(state["citations"]) == 1
    assert state["citations"][0].index == 1
    assert "informational only" in state["safety_notes"][0]


def test_invoke_with_no_chunks_returns_guardrail_answer() -> None:
    retriever = FakeRetriever([])
    llm = FakeLLM(response="I could not find that in the manuals.")
    state = _graph(retriever, llm).invoke("How do I rebuild a transmission?")

    assert state["confidence"] is None
    assert state["citations"] == []
    assert any("No matching documentation" in n for n in state["safety_notes"])
    assert state["context"] == ""


def test_invoke_passes_history_into_generated_messages() -> None:
    retriever = FakeRetriever([_chunk("context passage")])
    llm = FakeLLM()
    history = [
        ChatMessage(role="user", content="prior question"),
        ChatMessage(role="assistant", content="prior answer"),
    ]
    _graph(retriever, llm).invoke("current question", history=history)

    messages = llm.generated[0]
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1].content == "prior question"
    assert messages[2].content == "prior answer"
    assert "current question" in messages[3].content


def test_retrieve_state_builds_context_without_generating() -> None:
    retriever = FakeRetriever([_chunk("passage A")])
    llm = FakeLLM()
    prepared = _graph(retriever, llm).retrieve_state("query")
    assert prepared["context"].startswith("[1]")
    assert len(prepared["citations"]) == 1
    assert llm.generated == []


def test_compute_confidence() -> None:
    chunks = [_chunk("a", score=0.4), _chunk("b", score=0.86)]
    assert compute_confidence(chunks) == 0.86
    assert compute_confidence([]) is None


def test_build_safety_notes_empty_chunks() -> None:
    notes = build_safety_notes([])
    assert notes[0].startswith("No matching documentation")


def test_build_returns_compiled_graph() -> None:
    graph = _graph(FakeRetriever(), FakeLLM()).build()
    assert callable(getattr(graph, "invoke", None))
