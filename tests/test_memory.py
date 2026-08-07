"""Tests for durable conversation memory."""

from __future__ import annotations

import json

import pytest

from auto_rag.db.repositories import ConversationRepository
from auto_rag.errors import AgentError
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.models import Citation


@pytest.fixture
def memory(db) -> ConversationMemory:
    return ConversationMemory(ConversationRepository(db))


def test_get_or_create_creates_new_conversation(memory) -> None:
    conversation = memory.get_or_create()
    assert conversation.id is not None
    assert conversation.title == "New conversation"


def test_get_or_create_returns_existing_conversation(memory) -> None:
    created = memory.get_or_create()
    fetched = memory.get_or_create(created.id)
    assert fetched.id == created.id


def test_get_or_create_unknown_id_raises(memory) -> None:
    with pytest.raises(AgentError, match="not found"):
        memory.get_or_create(9999)


def test_history_roundtrip_user_and_assistant(memory) -> None:
    conversation = memory.get_or_create()
    memory.add_user_message(conversation.id, "How do I change oil?")
    memory.add_assistant_message(conversation.id, "Drain the pan.")
    history = memory.to_llm_history(conversation.id)
    assert [(m.role, m.content) for m in history] == [
        ("user", "How do I change oil?"),
        ("assistant", "Drain the pan."),
    ]


def test_assistant_citations_are_persisted(memory, db) -> None:
    conversation = memory.get_or_create()
    citations = [
        Citation(index=1, source="manual.pdf", score=0.9, page=4)
    ]
    memory.add_assistant_message(conversation.id, "Answer here", citations=citations)
    rows = ConversationRepository(db).list_messages(conversation.id)
    stored = json.loads(rows[-1].citations)
    assert stored == [
        {
            "index": 1,
            "source": "manual.pdf",
            "score": 0.9,
            "page": 4,
            "doc_type": "",
            "make": "",
            "model": "",
            "snippet": "",
        }
    ]


def test_history_respects_limit(memory, db) -> None:
    conversation = memory.get_or_create()
    for _ in range(15):
        memory.add_user_message(conversation.id, "turn")
        memory.add_assistant_message(conversation.id, "done")
    limited = ConversationMemory(ConversationRepository(db), history_limit=4)
    history = limited.to_llm_history(conversation.id)
    assert len(history) == 4
