"""Conversation memory backed by the SQLite conversation repository.

Bridges the persistence layer (Phase 2) and the LLM layer (Phase 5): stored
``messages`` rows become :class:`ChatMessage` history for the model, and each
turn is written back with its citations for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from auto_rag.db.models import Conversation
from auto_rag.db.repositories import ConversationRepository
from auto_rag.errors import AgentError
from auto_rag.llm.models import ChatMessage
from auto_rag.rag.models import Citation

logger = logging.getLogger(__name__)

__all__ = ["ConversationMemory"]


class ConversationMemory:
    """Load and persist chat history for a conversation."""

    def __init__(self, repository: ConversationRepository, history_limit: int = 10) -> None:
        self._repo = repository
        self.history_limit = history_limit

    def get_or_create(
        self,
        conversation_id: int | None = None,
        *,
        title: str | None = None,
    ) -> Conversation:
        """Return an existing conversation or create a new one.

        Raises:
            AgentError: When ``conversation_id`` does not exist.
        """
        if conversation_id is not None:
            conversation = self._repo.get(conversation_id)
            if conversation is None:
                raise AgentError(f"Conversation {conversation_id} not found")
            return conversation
        return self._repo.create(title=title or "New conversation")

    def to_llm_history(self, conversation_id: int) -> list[ChatMessage]:
        """Return the recent user/assistant turns as :class:`ChatMessage`."""
        rows = self._repo.list_messages(conversation_id, limit=self.history_limit)
        return [
            ChatMessage(role=row.role, content=row.content)
            for row in rows
            if row.role in ("user", "assistant")
        ]

    def add_user_message(self, conversation_id: int, content: str) -> None:
        self._repo.add_message(conversation_id, "user", content)

    def add_assistant_message(
        self,
        conversation_id: int,
        content: str,
        citations: list[Citation] | None = None,
    ) -> None:
        payload = [asdict(citation) for citation in citations] if citations else None
        self._repo.add_message(
            conversation_id, "assistant", content, citations=payload
        )
