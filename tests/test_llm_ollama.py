"""Tests for the Ollama LLM provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_rag.config import LLMConfig
from auto_rag.errors import LLMError
from auto_rag.llm.models import ChatMessage
from auto_rag.llm.ollama import OllamaLLM

_CONFIG = LLMConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    model="test-model",
    temperature=0.7,
    max_tokens=64,
    timeout_seconds=30.0,
)


def _full_response(content: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        message=SimpleNamespace(content=content),
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )


def _stream_chunk(content: str, *, done: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        message=SimpleNamespace(content=content),
        done=done,
        done_reason="stop" if done else None,
    )


class FakeClient:
    def __init__(
        self,
        *,
        chat_result: object | None = None,
        chat_error: Exception | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self._chat_result = chat_result
        self._chat_error = chat_error
        self._list_error = list_error
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._chat_error is not None:
            raise self._chat_error
        if self._chat_result is None:
            raise AssertionError("chat() called without a configured result")
        return self._chat_result

    def list(self):
        if self._list_error is not None:
            raise self._list_error
        return SimpleNamespace(models=[])


def _make_llm(client: FakeClient) -> OllamaLLM:
    return OllamaLLM(_CONFIG, client=client)


def test_generate_returns_content_and_usage() -> None:
    client = FakeClient(chat_result=_full_response())
    response = _make_llm(client).generate(
        [ChatMessage(role="user", content="Hi")]
    )
    assert response.content == "hello"
    assert response.model == "test-model"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.finish_reason == "stop"


def test_generate_sends_api_messages_and_default_options() -> None:
    client = FakeClient(chat_result=_full_response())
    llm = _make_llm(client)
    llm.generate([ChatMessage(role="system", content="Be brief")])
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "system", "content": "Be brief"}]
    assert call["stream"] is False
    assert call["options"] == {"temperature": 0.7, "num_predict": 64}


def test_generate_overrides_override_config_defaults() -> None:
    client = FakeClient(chat_result=_full_response())
    _make_llm(client).generate(
        [ChatMessage(role="user", content="Hi")],
        temperature=0.1,
        max_tokens=128,
    )
    assert client.calls[0]["options"] == {
        "temperature": 0.1,
        "num_predict": 128,
    }


def test_generate_raises_llm_error_on_backend_failure() -> None:
    client = FakeClient(chat_error=RuntimeError("boom"))
    with pytest.raises(LLMError, match="Ollama request failed"):
        _make_llm(client).generate([ChatMessage(role="user", content="Hi")])


def test_generate_raises_on_empty_content() -> None:
    client = FakeClient(chat_result=_full_response(content=""))
    with pytest.raises(LLMError, match="empty response"):
        _make_llm(client).generate([ChatMessage(role="user", content="Hi")])


def test_generate_stream_yields_content_parts() -> None:
    chunks = [_stream_chunk("Hel"), _stream_chunk("lo", done=True)]
    client = FakeClient(chat_result=chunks)
    tokens = list(
        _make_llm(client).generate_stream(
            [ChatMessage(role="user", content="Hi")]
        )
    )
    assert tokens == ["Hel", "lo"]
    assert client.calls[0]["stream"] is True


def test_generate_stream_raises_llm_error_on_backend_failure() -> None:
    client = FakeClient(chat_error=RuntimeError("boom"))
    with pytest.raises(LLMError, match="Ollama streaming failed"):
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
