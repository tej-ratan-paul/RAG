"""Tests for the LLM provider factory."""

from __future__ import annotations

import pytest

from auto_rag.config import LLMConfig
from auto_rag.errors import ConfigurationError
from auto_rag.llm.factory import build_llm
from auto_rag.llm.ollama import OllamaLLM
from auto_rag.llm.openai_compat import OpenAICompatLLM


def _config(provider: str) -> LLMConfig:
    return LLMConfig(provider=provider, model="test-model")


def test_build_ollama_provider() -> None:
    assert isinstance(build_llm(_config("ollama")), OllamaLLM)


@pytest.mark.parametrize("provider", ["openai", "openai_compat"])
def test_build_openai_compatible_provider(provider: str) -> None:
    assert isinstance(build_llm(_config(provider)), OpenAICompatLLM)


def test_provider_matching_is_case_insensitive() -> None:
    assert isinstance(build_llm(_config("OLLAMA")), OllamaLLM)
    assert isinstance(build_llm(_config("OpenAI")), OpenAICompatLLM)


def test_unknown_provider_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
        build_llm(_config("claude"))
