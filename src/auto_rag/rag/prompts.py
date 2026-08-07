"""Prompt templates for the RAG chat workflow.

Uses a LangChain :class:`ChatPromptTemplate` with a ``MessagesPlaceholder``
so conversation history is injected alongside the current question and the
retrieved context. Message converters bridge our :class:`ChatMessage` model
and LangChain's message types.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from auto_rag.llm.models import ChatMessage

__all__ = [
    "SYSTEM_PROMPT",
    "build_prompt_template",
    "to_langchain_messages",
    "from_langchain_messages",
]

SYSTEM_PROMPT: str = (
    "You are AutoRAG, an automotive repair assistant for service technicians "
    "and vehicle owners.\n\n"
    "Answer the user's question using ONLY the numbered repair context passages "
    "provided below.\n"
    "Rules:\n"
    "1. Base every claim on the context; cite passages inline as [1], [2], etc., "
    "matching the numbered passages.\n"
    "2. If the context does not contain the answer, say so explicitly and do not "
    "guess or invent specifications.\n"
    "3. Include practical safety notes where relevant (torque values, fluid types, "
    "warning signs, personal protective equipment).\n"
    "4. Keep repair procedures step-by-step and concise.\n"
    "5. Do not reference context that was not provided."
)


def build_prompt_template() -> ChatPromptTemplate:
    """Return the RAG chat prompt (system + history + question/context)."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nRepair context:\n{context}"),
        ]
    )


_ROLE_TO_LANGCHAIN = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

_LANGCHAIN_TYPE_TO_ROLE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


def to_langchain_messages(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    """Convert our :class:`ChatMessage` list into LangChain messages."""
    converted: list[BaseMessage] = []
    for message in messages:
        factory = _ROLE_TO_LANGCHAIN.get(message.role)
        if factory is None:
            raise ValueError(f"Unsupported chat role {message.role!r}")
        converted.append(factory(content=message.content))
    return converted


def from_langchain_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """Convert LangChain messages back into :class:`ChatMessage` instances."""
    converted: list[ChatMessage] = []
    for message in messages:
        role = _LANGCHAIN_TYPE_TO_ROLE.get(getattr(message, "type", ""))
        if role is None:
            raise ValueError(
                f"Unsupported message type {getattr(message, 'type', 'unknown')!r}"
            )
        content = str(message.content) if message.content is not None else ""
        converted.append(ChatMessage(role=role, content=content))
    return converted
