"""Ollama LLM provider.

Wraps the official ``ollama`` client (v0.6+ API). Generation parameters map
onto Ollama's ``options`` (``max_tokens`` -> ``num_predict``). Streaming uses
``chat(stream=True)``. All backend failures are raised as :class:`LLMError`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ollama import Client

from auto_rag.config import LLMConfig
from auto_rag.errors import LLMError
from auto_rag.llm.models import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)

__all__ = ["OllamaLLM"]


class OllamaLLM:
    """Chat completions against a local (or remote) Ollama server."""

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self._config = config
        self.model_name = config.model
        self._client = client or Client(
            host=config.base_url, timeout=config.timeout_seconds
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
        options = self._build_options(temperature, max_tokens)
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=payload,
                options=options,
                stream=False,
            )
        except Exception as exc:  # network, auth, server errors
            raise LLMError(f"Ollama request failed: {exc}") from exc

        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not content:
            raise LLMError("Ollama returned an empty response")
        return LLMResponse(
            content=content,
            model=response.model or self.model_name,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            finish_reason=response.done_reason,
        )

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = [message.to_api_dict() for message in messages]
        options = self._build_options(temperature, max_tokens)
        try:
            stream = self._client.chat(
                model=self.model_name,
                messages=payload,
                options=options,
                stream=True,
            )
            for chunk in stream:
                message = getattr(chunk, "message", None)
                content = getattr(message, "content", None) if message is not None else None
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    def ping(self) -> bool:
        """Return True when the Ollama server responds to ``list``."""
        try:
            self._client.list()
            return True
        except Exception:
            logger.debug("Ollama ping failed", exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_options(
        self, temperature: float | None, max_tokens: int | None
    ) -> dict[str, float | int]:
        """Merge caller overrides with config defaults into Ollama options."""
        options: dict[str, float | int] = {}
        if temperature is not None:
            options["temperature"] = temperature
        elif self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif self._config.max_tokens is not None:
            options["num_predict"] = self._config.max_tokens
        return options
