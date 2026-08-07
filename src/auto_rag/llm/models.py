"""LLM domain models shared across providers and callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ChatMessage", "LLMResponse", "ChatRole"]

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single conversational message exchanged with an LLM."""

    role: ChatRole
    content: str

    def to_api_dict(self) -> dict[str, str]:
        """Serialise into the role/content shape most chat APIs expect."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """A completed generation from an LLM."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
