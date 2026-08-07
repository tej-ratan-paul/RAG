"""Session-state helpers for the Streamlit UI.

Centralises the session keys, defaults, and the vehicle-context -> retrieval
filter translation so the app, the configuration page, and the tests agree on
a single source of truth. Widgets read their defaults here and write back
through the same accessors.
"""

from __future__ import annotations

import streamlit as st

from auto_rag.retrieval.models import RetrievalFilter

__all__ = [
    "VEHICLE_KEY",
    "CONVERSATION_KEY",
    "CONFIG_KEY",
    "TOP_K_KEY",
    "STREAM_KEY",
    "LAST_RESULT_KEY",
    "DEFAULT_VEHICLE",
    "DEFAULT_CONFIG",
    "VEHICLE_FIELDS",
    "init_session_state",
    "get_vehicle",
    "update_vehicle",
    "get_conversation_id",
    "set_conversation_id",
    "reset_conversation",
    "get_config",
    "update_config",
    "get_top_k",
    "set_top_k",
    "get_stream",
    "set_stream",
    "get_last_result",
    "set_last_result",
    "build_retrieval_filter",
]

VEHICLE_KEY = "ui.vehicle"
CONVERSATION_KEY = "ui.conversation_id"
CONFIG_KEY = "ui.config"
TOP_K_KEY = "ui.top_k"
STREAM_KEY = "ui.stream"
LAST_RESULT_KEY = "ui.last_result"

VEHICLE_FIELDS: tuple[str, ...] = ("make", "model", "year", "engine", "vin")
FILTER_FIELDS: tuple[str, ...] = ("make", "model", "year", "vin")

DEFAULT_VEHICLE: dict[str, object] = {
    "make": "",
    "model": "",
    "year": None,
    "engine": "",
    "vin": "",
}
DEFAULT_CONFIG: dict[str, object] = {
    "provider": "",
    "model": "",
    "base_url": "",
    "temperature": 0.1,
}


def init_session_state() -> None:
    """Seed every UI state key with its default, without clobbering values."""
    if VEHICLE_KEY not in st.session_state:
        st.session_state[VEHICLE_KEY] = dict(DEFAULT_VEHICLE)
    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = None
    if CONFIG_KEY not in st.session_state:
        st.session_state[CONFIG_KEY] = dict(DEFAULT_CONFIG)
    if TOP_K_KEY not in st.session_state:
        st.session_state[TOP_K_KEY] = 5
    if STREAM_KEY not in st.session_state:
        st.session_state[STREAM_KEY] = True
    if LAST_RESULT_KEY not in st.session_state:
        st.session_state[LAST_RESULT_KEY] = None


# ------------------------------------------------------------------ #
# Vehicle context
# ------------------------------------------------------------------ #
def get_vehicle() -> dict[str, object]:
    """Return a copy of the current vehicle context."""
    return dict(st.session_state[VEHICLE_KEY])


def update_vehicle(**fields: object) -> None:
    """Store the given vehicle fields, coercing ``year`` to int-or-None."""
    context = dict(st.session_state[VEHICLE_KEY])
    for field in VEHICLE_FIELDS:
        if field not in fields:
            continue
        value = fields[field]
        if field == "year":
            context[field] = int(value) if value else None
        else:
            context[field] = str(value).strip() if value else ""
    st.session_state[VEHICLE_KEY] = context


def build_retrieval_filter(vehicle: dict[str, object] | None = None) -> RetrievalFilter | None:
    """Translate a vehicle context onto a :class:`RetrievalFilter`.

    Only make/model/year/vin narrow retrieval; ``engine`` is kept for display
    context only. Returns None when no criterion is set.
    """
    context = dict(vehicle) if vehicle is not None else get_vehicle()
    criteria: dict[str, object] = {}
    for field in FILTER_FIELDS:
        value = context.get(field)
        if field == "year":
            if value:
                criteria[field] = int(value)
        elif value:
            criteria[field] = str(value).strip()
    return RetrievalFilter(**criteria) if criteria else None


# ------------------------------------------------------------------ #
# Conversation
# ------------------------------------------------------------------ #
def get_conversation_id() -> int | None:
    """Return the active conversation id (None means: start a new one)."""
    return st.session_state[CONVERSATION_KEY]


def set_conversation_id(value: int | None) -> None:
    st.session_state[CONVERSATION_KEY] = value


def reset_conversation() -> None:
    """Drop the active conversation link and the last-result panel."""
    st.session_state[CONVERSATION_KEY] = None
    st.session_state[LAST_RESULT_KEY] = None


# ------------------------------------------------------------------ #
# Configuration overrides
# ------------------------------------------------------------------ #
def get_config() -> dict[str, object]:
    """Return a copy of the LLM/retrieval override settings."""
    return dict(st.session_state[CONFIG_KEY])


def update_config(**fields: object) -> None:
    """Merge recognised overrides into the stored configuration."""
    config = dict(st.session_state[CONFIG_KEY])
    for field, value in fields.items():
        if field in config:
            config[field] = value
    st.session_state[CONFIG_KEY] = config


# ------------------------------------------------------------------ #
# Retrieval / rendering options
# ------------------------------------------------------------------ #
def get_top_k() -> int:
    return int(st.session_state[TOP_K_KEY])


def set_top_k(value: int) -> None:
    st.session_state[TOP_K_KEY] = int(value)


def get_stream() -> bool:
    return bool(st.session_state[STREAM_KEY])


def set_stream(value: bool) -> None:
    st.session_state[STREAM_KEY] = bool(value)


def get_last_result():
    """Return the most recent :class:`RAGResult` (or None)."""
    return st.session_state[LAST_RESULT_KEY]


def set_last_result(result) -> None:
    st.session_state[LAST_RESULT_KEY] = result
