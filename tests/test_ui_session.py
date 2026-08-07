"""Unit tests for the UI session-state helpers (patched session_state)."""

from __future__ import annotations

import pytest
import streamlit as st

from auto_rag.ui import session as session_module
from auto_rag.ui.session import (
    CONFIG_KEY,
    CONVERSATION_KEY,
    LAST_RESULT_KEY,
    STREAM_KEY,
    TOP_K_KEY,
    VEHICLE_KEY,
    build_retrieval_filter,
    get_config,
    get_conversation_id,
    get_stream,
    get_top_k,
    get_vehicle,
    init_session_state,
    reset_conversation,
    set_conversation_id,
    set_last_result,
    set_stream,
    set_top_k,
    update_config,
    update_vehicle,
)


class FakeSession(dict):
    pass


@pytest.fixture
def fake_session(monkeypatch) -> FakeSession:
    store = FakeSession()
    monkeypatch.setattr(st, "session_state", store)
    return store


def test_init_session_state_seeds_defaults(fake_session) -> None:
    init_session_state()
    assert fake_session[VEHICLE_KEY] == session_module.DEFAULT_VEHICLE
    assert fake_session[CONVERSATION_KEY] is None
    assert fake_session[CONFIG_KEY] == session_module.DEFAULT_CONFIG
    assert fake_session[TOP_K_KEY] == 5
    assert fake_session[STREAM_KEY] is True
    assert fake_session[LAST_RESULT_KEY] is None


def test_init_session_state_preserves_existing(fake_session) -> None:
    fake_session[VEHICLE_KEY] = {"make": "Toyota"}
    init_session_state()
    assert fake_session[VEHICLE_KEY]["make"] == "Toyota"


def test_vehicle_get_update_and_coercion(fake_session) -> None:
    init_session_state()
    update_vehicle(make="Toyota", model="Camry", year="2018", engine="2.5L", vin="ABC123")
    vehicle = get_vehicle()
    assert vehicle == {
        "make": "Toyota",
        "model": "Camry",
        "year": 2018,
        "engine": "2.5L",
        "vin": "ABC123",
    }
    update_vehicle(year="")
    assert get_vehicle()["year"] is None


def test_vehicle_update_ignores_unknown_fields(fake_session) -> None:
    init_session_state()
    update_vehicle(make="Honda", colour="red")
    assert "colour" not in get_vehicle()


def test_conversation_id_get_set_reset(fake_session) -> None:
    init_session_state()
    assert get_conversation_id() is None
    set_conversation_id(7)
    assert get_conversation_id() == 7
    reset_conversation()
    assert get_conversation_id() is None
    assert fake_session[LAST_RESULT_KEY] is None


def test_config_get_update(fake_session) -> None:
    init_session_state()
    update_config(model="llama3.2", base_url="http://127.0.0.1:11434")
    config = get_config()
    assert config["model"] == "llama3.2"
    assert config["base_url"] == "http://127.0.0.1:11434"
    update_config(unknown_field=1)
    assert "unknown_field" not in get_config()


def test_top_k_and_stream_defaults_and_setters(fake_session) -> None:
    init_session_state()
    assert get_top_k() == 5
    assert get_stream() is True
    set_top_k(8)
    set_stream(False)
    assert get_top_k() == 8
    assert get_stream() is False


def test_set_last_result(fake_session) -> None:
    init_session_state()
    set_last_result("result")
    assert fake_session[LAST_RESULT_KEY] == "result"


def test_build_retrieval_filter_from_context() -> None:
    retrieval_filter = build_retrieval_filter(
        {"make": "Toyota", "model": "Camry", "year": 2018, "engine": "2.5L", "vin": ""}
    )
    assert retrieval_filter is not None
    assert retrieval_filter.make == "Toyota"
    assert retrieval_filter.model == "Camry"
    assert retrieval_filter.year == 2018
    assert retrieval_filter.vin is None
    assert retrieval_filter.as_dict() == {"make": "Toyota", "model": "Camry", "year": 2018}


def test_build_retrieval_filter_none_when_empty() -> None:
    assert build_retrieval_filter({"make": "", "model": "", "year": None, "vin": ""}) is None


def test_build_retrieval_filter_year_only() -> None:
    retrieval_filter = build_retrieval_filter({"year": "2020", "make": "", "model": "", "vin": ""})
    assert retrieval_filter is not None
    assert retrieval_filter.year == 2020
    assert retrieval_filter.as_dict() == {"year": 2020}
