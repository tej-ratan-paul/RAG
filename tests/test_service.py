"""Integration tests for the RAG service over the real retrieval + DB layers."""

from __future__ import annotations

import pytest

from auto_rag.config import RetrievalConfig
from auto_rag.db.repositories import ConversationRepository
from auto_rag.errors import AgentError
from auto_rag.ingestion.chunking import Chunk
from auto_rag.llm.models import ChatMessage, LLMResponse
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.service import RAGService
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.reranker import NoopReranker
from auto_rag.retrieval.retriever import Retriever


class FakeLLM:
    """Streaming-capable LLM double for service tests."""

    def __init__(self, response: str = "Torque to 25 Nm [1].") -> None:
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
        del temperature, max_tokens
        self.generated.append(messages)
        words = self._response.split(" ")
        # Emulate real token streaming: continuation tokens carry a leading space.
        tokens = [words[0]] + [f" {word}" for word in words[1:]]
        return iter(tokens)


def _chunk_obj(text: str, index: int, **meta) -> Chunk:
    metadata = {
        "chunk_index": index,
        "title": "x",
        "doc_type": "service_manual",
        "make": "",
        "model": "",
        "year": "",
        "engine": "",
        "vin": "",
        **meta,
    }
    return Chunk(text=text, metadata=metadata)


def _build_service(db, vector_store, llm: FakeLLM) -> RAGService:
    bm25 = BM25Index(vector_store.get_all_chunks())
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=vector_store.provider,
        config=RetrievalConfig(hybrid_search=False, mmr=False, rerank=False),
        bm25_index=bm25,
        reranker=NoopReranker(),
    )
    memory = ConversationMemory(ConversationRepository(db))
    return RAGService(retriever, llm, memory)


def test_ask_returns_structured_result(db, vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("Torque the brake caliper bolts to 25 Nm", 0)],
        source="manual.pdf",
    )
    llm = FakeLLM(response="Torque to 25 Nm [1].")
    service = _build_service(db, vector_store, llm)

    result = service.ask("What torque for brake calipers?")

    assert result.query == "What torque for brake calipers?"
    assert result.answer == "Torque to 25 Nm [1]."
    assert len(result.sources) == 1
    assert result.sources[0].source == "manual.pdf"
    assert result.confidence is not None
    assert result.safety_notes
    assert result.conversation_id is not None


def test_ask_persists_exchange(db, vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("Inspect the serpentine belt for cracks", 0)],
        source="manual.pdf",
    )
    service = _build_service(db, vector_store, FakeLLM())
    result = service.ask("How do I check the belt?")

    repo = ConversationRepository(db)
    messages = repo.list_messages(result.conversation_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "How do I check the belt?"),
        ("assistant", service.last_result.answer),
    ]


def test_ask_reuses_conversation_and_injects_history(db, vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("brake pad minimum thickness is 3 mm", 0)],
        source="manual.pdf",
    )
    llm = FakeLLM(response="3 mm [1].")
    service = _build_service(db, vector_store, llm)

    first = service.ask("What is the minimum brake pad thickness?")
    service.ask(
        "And the brake fluid?", conversation_id=first.conversation_id
    )

    messages = llm.generated[1]
    assert [(m.role, m.content) for m in messages] == [
        ("system", messages[0].content),
        ("user", "What is the minimum brake pad thickness?"),
        ("assistant", "3 mm [1]."),
        ("user", messages[3].content),
    ]
    assert messages[3].content.startswith("Question: And the brake fluid?")


def test_ask_unknown_conversation_raises(db, vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("replace the air filter", 0)],
        source="manual.pdf",
    )
    service = _build_service(db, vector_store, FakeLLM())
    with pytest.raises(AgentError, match="not found"):
        service.ask("air filter?", conversation_id=4242)


def test_ask_stream_yields_tokens_and_sets_last_result(db, vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("Drain plug torque is 35 Nm", 0)],
        source="manual.pdf",
    )
    llm = FakeLLM(response="Drain plug to 35 Nm [1].")
    service = _build_service(db, vector_store, llm)

    tokens = list(service.ask_stream("Oil drain plug torque?"))

    assert tokens == ["Drain", " plug", " to", " 35", " Nm", " [1]."]
    result = service.last_result
    assert result is not None
    assert result.answer == "Drain plug to 35 Nm [1]."
    assert result.conversation_id is not None

    repo = ConversationRepository(db)
    messages = repo.list_messages(result.conversation_id)
    assert len(messages) == 2
    assert messages[1].role == "assistant"
    assert messages[1].citations is not None
