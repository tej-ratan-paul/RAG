"""LLM provider protocol.

Every provider (Ollama, OpenAI-compatible, test doubles) implements this
minimal interface so the RAG layer never depends on a concrete SDK.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from auto_rag.llm.models import ChatMessage, LLMResponse

__all__ = ["LLM"]


class LLM(Protocol):
    """Uniform interface for chat completion providers."""

    model_name: str

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a complete response for ``messages``."""
        ...

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield response content incrementally as it is produced."""
        ...

    def ping(self) -> bool:
        """Return True when the backend is reachable."""
        ...
