"""Tests for the OpenAI-compatible LLM provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_rag.config import LLMConfig
from auto_rag.errors import LLMError
from auto_rag.llm.models import ChatMessage
from auto_rag.llm.openai_compat import OpenAICompatLLM

_CONFIG = LLMConfig(
    provider="openai",
    base_url="http://localhost:8000/v1",
    model="test-model",
    temperature=0.5,
    max_tokens=128,
    timeout_seconds=30.0,
)


def _completion(content: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )


def _stream_chunk(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("create() called without a configured result")
        return self._result


class FakeModels:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[dict] = []

    def list(self):
        self.calls.append({})
        if self._error is not None:
            raise self._error
        return SimpleNamespace(data=[])


class FakeClient:
    def __init__(
        self,
        *,
        create_result: object | None = None,
        create_error: Exception | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.completions = FakeCompletions(create_result, create_error)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = FakeModels(list_error)


def _make_llm(client: FakeClient) -> OpenAICompatLLM:
    return OpenAICompatLLM(_CONFIG, client=client)


def test_generate_returns_content_and_usage() -> None:
    response = _make_llm(FakeClient(create_result=_completion())).generate(
        [ChatMessage(role="user", content="Hi")]
    )
    assert response.content == "hello"
    assert response.model == "test-model"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 4
    assert response.finish_reason == "stop"


def test_generate_sends_payload_and_config_defaults() -> None:
    client = FakeClient(create_result=_completion())
    _make_llm(client).generate([ChatMessage(role="user", content="Hi")])
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "Hi"}]
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 128
    assert call["stream"] is False


def test_generate_overrides_override_config_defaults() -> None:
    client = FakeClient(create_result=_completion())
    _make_llm(client).generate(
        [ChatMessage(role="user", content="Hi")],
        temperature=0.9,
        max_tokens=256,
    )
    call = client.completions.calls[0]
    assert call["temperature"] == 0.9
    assert call["max_tokens"] == 256


def test_generate_raises_llm_error_on_backend_failure() -> None:
    client = FakeClient(create_error=RuntimeError("boom"))
    with pytest.raises(LLMError, match="OpenAI-compatible request failed"):
        _make_llm(client).generate([ChatMessage(role="user", content="Hi")])


def test_generate_raises_on_empty_content() -> None:
    client = FakeClient(create_result=_completion(content=""))
    with pytest.raises(LLMError, match="empty response"):
        _make_llm(client).generate([ChatMessage(role="user", content="Hi")])


def test_generate_stream_yields_content_parts() -> None:
    client = FakeClient(create_result=[_stream_chunk("Hel"), _stream_chunk("lo")])
    tokens = list(
        _make_llm(client).generate_stream(
            [ChatMessage(role="user", content="Hi")]
        )
    )
    assert tokens == ["Hel", "lo"]
    assert client.completions.calls[0]["stream"] is True


def test_generate_stream_raises_llm_error_on_backend_failure() -> None:
    client = FakeClient(create_error=RuntimeError("boom"))
    with pytest.raises(LLMError, match="OpenAI-compatible streaming failed"):
        list(
            _make_llm(client).generate_stream(
                [ChatMessage(role="user", content="Hi")]
            )
        )


def test_ping_true_when_server_responds() -> None:
    assert _make_llm(FakeClient()).ping() is True


def test_ping_false_when_server_unreachable() -> None:
    client = FakeClient(list_error=ConnectionError("refused"))
    assert _make_llm(client).ping() is False
