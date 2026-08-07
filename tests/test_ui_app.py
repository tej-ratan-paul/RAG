"""End-to-end tests for the Streamlit UI via streamlit.testing AppTest.

A fake RAGService/conversation bundle is injected through session state so the
app runs headlessly without the real retriever/LLM stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit.testing.v1 as stt

from auto_rag.db.models import Conversation, Message
from auto_rag.errors import AutoRAGError
from auto_rag.rag.models import Citation, RAGResult

APP_PATH = Path(__file__).resolve().parent.parent / "src" / "auto_rag" / "ui" / "app.py"


class FakeConversations:
    """In-memory conversation repository used by the fake bundle."""

    def __init__(self) -> None:
        self.conversations: list[Conversation] = [Conversation(id=1, title="Brake diagnosis")]
        self.messages: dict[int, list[Message]] = {
            1: [
                Message(
                    conversation_id=1,
                    role="user",
                    content="Front brakes grind when stopping.",
                ),
                Message(
                    conversation_id=1,
                    role="assistant",
                    content="Inspect the brake pads.",
                    citations=json.dumps(
                        [
                            {
                                "index": 1,
                                "source": "brakes.pdf",
                                "score": 0.91,
                                "page": 4,
                                "doc_type": "repair_manual",
                                "make": "Toyota",
                                "model": "Camry",
                                "snippet": "Minimum pad thickness is 3 mm.",
                            }
                        ]
                    ),
                ),
            ]
        }
        self._next_id = 2

    def list_all(self) -> list[Conversation]:
        return list(self.conversations)

    def get(self, conversation_id: int) -> Conversation | None:
        for conversation in self.conversations:
            if conversation.id == conversation_id:
                return conversation
        return None

    def list_messages(self, conversation_id: int) -> list[Message]:
        return list(self.messages.get(conversation_id, []))

    def clear_messages(self, conversation_id: int) -> None:
        self.messages[conversation_id] = []


class FakeService:
    """Streaming-capable RAG service double that persists into FakeConversations."""

    def __init__(
        self,
        repo: FakeConversations,
        answer: str = "Torque to 25 Nm [1].",
        error: str | None = None,
    ) -> None:
        self.repo = repo
        self.answer = answer
        self.error = error
        self.last_result: RAGResult | None = None
        self.asked: list[tuple[str, int | None]] = []

    def _conversation_id(self, conversation_id: int | None) -> int:
        if conversation_id is None:
            conversation_id = self.repo._next_id
            self.repo._next_id += 1
            self.repo.conversations.append(
                Conversation(id=conversation_id, title="New conversation")
            )
            self.repo.messages[conversation_id] = []
        return conversation_id

    def _complete(self, question: str, conversation_id: int | None) -> RAGResult:
        if self.error:
            raise AutoRAGError(self.error)
        self.asked.append((question, conversation_id))
        conv_id = self._conversation_id(conversation_id)
        result = RAGResult(
            query=question,
            answer=self.answer,
            sources=[
                Citation(
                    index=1,
                    source="manual.pdf",
                    score=0.95,
                    page=3,
                    doc_type="repair_manual",
                    make="Toyota",
                    model="Camry",
                    snippet="Torque to 25 Nm.",
                )
            ],
            confidence=0.92,
            safety_notes=["Disconnect the battery before working."],
            conversation_id=conv_id,
            model="fake-model",
        )
        self.last_result = result
        self.repo.messages[conv_id].append(
            Message(conversation_id=conv_id, role="user", content=question)
        )
        self.repo.messages[conv_id].append(
            Message(
                conversation_id=conv_id,
                role="assistant",
                content=result.answer,
                citations=json.dumps(
                    [
                        {
                            "index": 1,
                            "source": "manual.pdf",
                            "score": 0.95,
                            "page": 3,
                            "doc_type": "repair_manual",
                            "make": "Toyota",
                            "model": "Camry",
                            "snippet": "Torque to 25 Nm.",
                        }
                    ]
                ),
            )
        )
        return result

    def ask(
        self, question: str, *, conversation_id=None, retrieval_filter=None, top_k=None, title=None
    ) -> RAGResult:
        return self._complete(question, conversation_id)

    def ask_stream(self, question: str, *, conversation_id=None, retrieval_filter=None, top_k=None):
        yield from self.answer.split(" ")
        self._complete(question, conversation_id)


@pytest.fixture
def bundle() -> object:
    repo = FakeConversations()
    return type(
        "Bundle",
        (),
        {"service": FakeService(repo), "conversations": repo, "db": None},
    )()


@pytest.fixture
def failing_bundle() -> object:
    repo = FakeConversations()
    return type(
        "Bundle",
        (),
        {
            "service": FakeService(repo, error="Backend unreachable"),
            "conversations": repo,
            "db": None,
        },
    )()


def _make_app(bundle: object) -> stt.AppTest:
    at = stt.AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["ui.bundle"] = bundle
    return at


def _button(at: stt.AppTest, key: str):
    for element in (*at.button, *at.sidebar.button):
        if element.key == key:
            return element
    raise AssertionError(f"No button with key {key!r}")


def test_app_initial_render(bundle) -> None:
    at = _make_app(bundle)
    at.run()
    assert at.title[0].value == "AutoRAG Repair Assistant"
    assert len(at.tabs) == 2
    assert at.chat_input
    labels = [c.value for c in at.sidebar.text_input]
    assert any(label is not None for label in labels)


def test_app_streams_answer_and_persists(bundle) -> None:
    at = _make_app(bundle)
    at.run()
    assert bundle.service.last_result is None

    at.chat_input[0].set_value("What torque for the caliper bolts?").run()

    assert bundle.service.last_result is not None
    assert bundle.service.last_result.answer == "Torque to 25 Nm [1]."
    assert at.session_state["ui.last_result"].answer == "Torque to 25 Nm [1]."
    assert at.session_state["ui.conversation_id"] == 2
    names = [message.name for message in at.chat_message]
    assert names == ["user", "assistant"]
    rendered = [m.value for m in at.chat_message[1].markdown]
    assert any("Torque to 25 Nm [1]." in value for value in rendered)


def test_app_non_stream_uses_ask(bundle) -> None:
    at = _make_app(bundle)
    at.session_state["ui.stream"] = False
    at.run()
    at.chat_input[0].set_value("Brake pad thickness?").run()
    assert bundle.service.asked
    assert bundle.service.asked[0][0] == "Brake pad thickness?"
    assert at.session_state["ui.last_result"].answer == "Torque to 25 Nm [1]."


def test_app_renders_existing_history_and_citations(bundle) -> None:
    at = _make_app(bundle)
    at.session_state["ui.conversation_id"] = 1
    at.run()
    names = [message.name for message in at.chat_message]
    assert names == ["user", "assistant"]
    assert any("Front brakes grind" in m.value for m in at.chat_message[0].markdown)
    assert any("brakes.pdf" in value.value for value in at.caption)


def test_app_sources_panel_after_answer(bundle) -> None:
    at = _make_app(bundle)
    at.run()
    at.chat_input[0].set_value("Torque?").run()
    assert at.expander
    expander_markdown = [m.value for m in at.expander[0].markdown]
    assert any("manual.pdf" in value for value in expander_markdown)
    assert any("Confidence" in value for value in expander_markdown)


def test_app_new_conversation_button(bundle) -> None:
    at = _make_app(bundle)
    at.session_state["ui.conversation_id"] = 1
    at.run()
    _button(at, "ui_conv_new").click().run()
    assert at.session_state["ui.conversation_id"] is None
    assert at.chat_message == []


def test_app_clear_conversation_button(bundle) -> None:
    at = _make_app(bundle)
    at.session_state["ui.conversation_id"] = 1
    at.run()
    assert len(bundle.conversations.list_messages(1)) == 2
    _button(at, "ui_conv_clear").click().run()
    assert bundle.conversations.list_messages(1) == []


def test_app_shows_error_when_llm_fails(failing_bundle) -> None:
    at = _make_app(failing_bundle)
    at.run()
    at.chat_input[0].set_value("Diagnose this.").run()
    assert any("Backend unreachable" in error.value for error in at.error)


def test_app_config_page_applies_overrides(bundle) -> None:
    at = _make_app(bundle)
    at.run()
    model_input = next(element for element in at.text_input if element.key == "ui_config_model")
    base_url_input = next(
        element for element in at.text_input if element.key == "ui_config_base_url"
    )
    model_input.set_value("llama3.2")
    base_url_input.set_value("http://localhost:9999")
    _button(at, "ui_config_apply").click().run()
    config = at.session_state["ui.config"]
    assert config["model"] == "llama3.2"
    assert config["base_url"] == "http://localhost:9999"
    assert at.success
