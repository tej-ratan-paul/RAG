"""OpenAI-compatible chat completions provider.

Talks to any server exposing the OpenAI ``/chat/completions`` API: OpenAI,
Ollama's OpenAI compatibility layer, vLLM, LM Studio, and so on. When no
``api_key`` is configured a placeholder is sent, which local servers accept.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from auto_rag.config import LLMConfig
from auto_rag.errors import LLMError
from auto_rag.llm.models import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatLLM"]

_PLACEHOLDER_API_KEY: str = "not-needed"


class OpenAICompatLLM:
    """Chat completions against an OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self._config = config
        self.model_name = config.model
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=config.api_key or _PLACEHOLDER_API_KEY,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
            )

    # ------------------------------------------------------------------ #
    # LLM protocol
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = [message.to_api_dict() for message in messages]
        temperature = self._config.temperature if temperature is None else temperature
        max_tokens = self._config.max_tokens if max_tokens is None else max_tokens
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:  # network, auth, server errors
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice else None
        content = getattr(message, "content", None) if message is not None else None
        if not content:
            raise LLMError("OpenAI-compatible endpoint returned an empty response")
        usage = completion.usage
        return LLMResponse(
            content=content,
            model=completion.model or self.model_name,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            finish_reason=choice.finish_reason if choice is not None else None,
        )

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = [message.to_api_dict() for message in messages]
        temperature = self._config.temperature if temperature is None else temperature
        max_tokens = self._config.max_tokens if max_tokens is None else max_tokens
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                delta = choice.delta if choice is not None else None
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise LLMError(f"OpenAI-compatible streaming failed: {exc}") from exc

    def ping(self) -> bool:
        """Return True when the endpoint responds to ``models.list``."""
        try:
            self._client.models.list()
            return True
        except Exception:
            logger.debug("OpenAI-compatible ping failed", exc_info=True)
            return False
