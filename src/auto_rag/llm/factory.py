"""LLM provider factory."""

from __future__ import annotations

from auto_rag.config import LLMConfig
from auto_rag.errors import ConfigurationError
from auto_rag.llm.base import LLM
from auto_rag.llm.ollama import OllamaLLM
from auto_rag.llm.openai_compat import OpenAICompatLLM

__all__ = ["build_llm"]

_SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "openai", "openai_compat")


def build_llm(config: LLMConfig) -> LLM:
    """Build the configured LLM provider.

    Args:
        config: LLM settings (``provider`` selects the backend).

    Returns:
        An :class:`LLM` implementation ready to generate.

    Raises:
        ConfigurationError: When ``config.provider`` is unsupported.
    """
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return OllamaLLM(config)
    if provider in ("openai", "openai_compat"):
        return OpenAICompatLLM(config)
    raise ConfigurationError(
        f"Unsupported LLM provider {config.provider!r}; "
        f"choose from {_SUPPORTED_PROVIDERS}"
    )
