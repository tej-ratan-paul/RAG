"""Tests for the RAG prompt templates and message converters."""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from auto_rag.llm.models import ChatMessage
from auto_rag.rag.prompts import (
    build_prompt_template,
    from_langchain_messages,
    to_langchain_messages,
)


def test_prompt_template_renders_context_and_history() -> None:
    template = build_prompt_template()
    rendered = template.invoke(
        {
            "question": "Torque spec?",
            "context": "[1] (manual.pdf) 25 Nm",
            "history": to_langchain_messages(
                [
                    ChatMessage(role="user", content="Hello"),
                    ChatMessage(role="assistant", content="Hi!"),
                ]
            ),
        }
    )
    texts = [m.content for m in rendered.messages]
    joined = " ".join(texts)
    assert "Torque spec?" in joined
    assert "[1] (manual.pdf) 25 Nm" in joined
    assert "Hello" in joined
    assert "Hi!" in joined


def test_prompt_template_system_instructions_present() -> None:
    template = build_prompt_template()
    rendered = template.invoke(
        {"question": "q", "context": "c", "history": []}
    )
    assert "cite passages inline" in rendered.messages[0].content


def test_message_roundtrip_preserves_role_and_content() -> None:
    original = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    ]
    converted = from_langchain_messages(to_langchain_messages(original))
    assert converted == original


def test_from_langchain_rejects_unknown_message_type() -> None:
    tool_message = ToolMessage(content="tool output", tool_call_id="1")
    with pytest.raises(ValueError, match="Unsupported message type"):
        from_langchain_messages([tool_message])
