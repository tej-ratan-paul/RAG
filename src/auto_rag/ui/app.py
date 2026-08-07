"""AutoRAG Streamlit application.

A chat front end over the Phase 6 RAG service: streaming answers, source
citations, confidence and safety notes, a vehicle-context sidebar that narrows
retrieval, conversation history management, and a configuration page.

Run with::

    streamlit run src/auto_rag/ui/app.py
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from auto_rag.errors import AutoRAGError
from auto_rag.rag.citations import format_citations
from auto_rag.rag.models import Citation, RAGResult
from auto_rag.ui.config_page import render_config_page
from auto_rag.ui.service import UIServiceBundle, build_bundle
from auto_rag.ui.session import (
    build_retrieval_filter,
    get_config,
    get_conversation_id,
    get_last_result,
    get_stream,
    get_top_k,
    get_vehicle,
    init_session_state,
    reset_conversation,
    set_conversation_id,
    set_last_result,
    set_stream,
    set_top_k,
    update_vehicle,
)

__all__ = ["main", "run"]

_BUNDLE_KEY = "ui.bundle"
_PENDING_SELECT_KEY = "ui.pending_conv_select"


# ------------------------------------------------------------------ #
# Service bootstrap
# ------------------------------------------------------------------ #
def _bundle() -> UIServiceBundle:
    """Return an injected bundle (tests) or a cached real bundle."""
    injected = st.session_state.get(_BUNDLE_KEY)
    if injected is not None:
        return injected
    config = get_config()
    return build_bundle(
        provider=str(config["provider"]) or None,
        model=str(config["model"]) or None,
        base_url=str(config["base_url"]) or None,
        temperature=float(config["temperature"]),
    )


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
def _year_from_text(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def _conversation_label(bundle: UIServiceBundle, conversation_id: int | None) -> str:
    """Render the selectbox label for a conversation id."""
    if conversation_id is None:
        return "New conversation"
    for conversation in bundle.conversations.list_all():
        if conversation.id == conversation_id:
            return f"#{conversation.id} {conversation.title}"
    return "New conversation"


def _consume_pending_conversation_select() -> None:
    """Apply a queued selectbox value before its widget is instantiated.

    Widget keys cannot be modified after their widget is created in the current
    run, so code that changes the conversation programmatically queues the new
    selectbox label here; the sidebar applies it at the top of the next run.
    """
    if _PENDING_SELECT_KEY not in st.session_state:
        return
    st.session_state["ui_conv_select"] = st.session_state[_PENDING_SELECT_KEY]
    del st.session_state[_PENDING_SELECT_KEY]


def _render_sidebar(bundle: UIServiceBundle) -> None:
    st.sidebar.title("AutoRAG Repair Assistant")

    _consume_pending_conversation_select()

    vehicle = get_vehicle()
    st.sidebar.subheader("Vehicle context")
    make = st.sidebar.text_input("Make", value=str(vehicle["make"]), key="ui_vehicle_make")
    model = st.sidebar.text_input("Model", value=str(vehicle["model"]), key="ui_vehicle_model")
    year_raw = st.sidebar.text_input(
        "Year", value=str(vehicle["year"] or ""), placeholder="e.g. 2018", key="ui_vehicle_year"
    )
    engine = st.sidebar.text_input("Engine", value=str(vehicle["engine"]), key="ui_vehicle_engine")
    vin = st.sidebar.text_input("VIN", value=str(vehicle["vin"]), key="ui_vehicle_vin")

    update_vehicle(
        make=make,
        model=model,
        year=_year_from_text(year_raw),
        engine=engine,
        vin=vin,
    )

    retrieval_filter = build_retrieval_filter()
    if retrieval_filter is not None and retrieval_filter.active:
        summary = ", ".join(f"{k}={v}" for k, v in retrieval_filter.as_dict().items())
        st.sidebar.caption(f"Retrieval filtered by: {summary}")

    st.sidebar.subheader("Conversation")
    conversations = bundle.conversations.list_all()
    labels = [f"#{c.id} {c.title}" for c in conversations]
    options = ["New conversation", *labels]
    current_index = 0
    if get_conversation_id() is not None:
        for position, conversation in enumerate(conversations, start=1):
            if conversation.id == get_conversation_id():
                current_index = position
                break
    # Once the widget key exists (it is set by _consume_pending_conversation_select),
    # pass index=None so Streamlit uses the stored value instead of warning about
    # both a default value and a Session State value.
    default_index = current_index if "ui_conv_select" not in st.session_state else None
    selected = st.sidebar.selectbox(
        "Active conversation", options, index=default_index, key="ui_conv_select"
    )
    if selected == "New conversation":
        set_conversation_id(None)
    else:
        conversation_id = int(selected.split(" ", 1)[0][1:])
        if conversation_id != get_conversation_id():
            set_conversation_id(conversation_id)

    new_col, clear_col = st.sidebar.columns(2)
    if new_col.button("New", key="ui_conv_new"):
        reset_conversation()
        st.session_state[_PENDING_SELECT_KEY] = "New conversation"
        st.rerun()
    if clear_col.button("Clear", key="ui_conv_clear"):
        conversation_id = get_conversation_id()
        if conversation_id is not None:
            bundle.conversations.clear_messages(conversation_id)
        st.rerun()

    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider(
        "Top-k sources", min_value=1, max_value=10, value=get_top_k(), key="ui_top_k"
    )
    set_top_k(top_k)
    streaming = st.sidebar.toggle("Stream answers", value=get_stream(), key="ui_stream")
    set_stream(streaming)


# ------------------------------------------------------------------ #
# Chat rendering
# ------------------------------------------------------------------ #
def _citation_from_payload(item: dict[str, Any]) -> Citation:
    return Citation(
        index=int(item.get("index", 0)),
        source=item.get("source", ""),
        score=float(item.get("score") or 0.0),
        page=item.get("page"),
        doc_type=item.get("doc_type", ""),
        make=item.get("make", ""),
        model=item.get("model", ""),
        snippet=item.get("snippet", ""),
    )


def _render_history(bundle: UIServiceBundle) -> None:
    conversation_id = get_conversation_id()
    if conversation_id is None:
        return
    for message in bundle.conversations.list_messages(conversation_id):
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role == "assistant" and message.citations:
                try:
                    payload = json.loads(message.citations)
                    citations = [_citation_from_payload(item) for item in payload]
                    if citations:
                        st.caption(format_citations(citations))
                except (TypeError, ValueError):
                    pass


def _render_source_panel(result: RAGResult | None) -> None:
    if result is None:
        return
    with st.expander(f"Sources ({len(result.sources)})", expanded=False):
        if result.sources:
            st.markdown(format_citations(result.sources))
        else:
            st.markdown("No sources were retrieved for this answer.")
        if result.confidence is not None:
            st.markdown(f"Confidence: **{result.confidence:.2f}**")
        if result.safety_notes:
            st.markdown("**Safety notes:**")
            for note in result.safety_notes:
                st.markdown(f"- {note}")


def _handle_prompt(bundle: UIServiceBundle, prompt: str) -> None:
    retrieval_filter = build_retrieval_filter()
    top_k = get_top_k()
    conversation_id = get_conversation_id()
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        if get_stream():
            with st.chat_message("assistant"):
                st.write_stream(
                    bundle.service.ask_stream(
                        prompt,
                        conversation_id=conversation_id,
                        retrieval_filter=retrieval_filter,
                        top_k=top_k,
                    )
                )
        else:
            result = bundle.service.ask(
                prompt,
                conversation_id=conversation_id,
                retrieval_filter=retrieval_filter,
                top_k=top_k,
            )
            with st.chat_message("assistant"):
                st.markdown(result.answer)
    except AutoRAGError as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
        return
    result = bundle.service.last_result
    if result is not None:
        set_conversation_id(result.conversation_id)
        set_last_result(result)
        st.session_state[_PENDING_SELECT_KEY] = _conversation_label(bundle, result.conversation_id)
    st.rerun()


def _render_chat_tab(bundle: UIServiceBundle) -> None:
    _render_history(bundle)
    _render_source_panel(get_last_result())
    prompt = st.chat_input("Ask about your vehicle, e.g. 'P0300 misfire diagnosis'")
    if prompt:
        _handle_prompt(bundle, prompt)


# ------------------------------------------------------------------ #
# Entrypoint
# ------------------------------------------------------------------ #
def main() -> None:
    st.set_page_config(page_title="AutoRAG Repair Assistant", page_icon=":wrench:", layout="wide")
    init_session_state()
    bundle = _bundle()

    _render_sidebar(bundle)

    chat_tab, config_tab = st.tabs(["Chat", "Configuration"])
    with chat_tab:
        _render_chat_tab(bundle)
    with config_tab:
        render_config_page()


def run() -> None:
    """Launch the app via ``streamlit run`` (console-script entrypoint)."""
    import sys
    from pathlib import Path

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
