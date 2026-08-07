"""Configuration page rendered in the main "Configuration" tab.

Lets the operator override the LLM provider, model, base URL, and temperature
for the current session without restarting the app. Applying the form stores
the overrides in session state; the app then builds a fresh bundle keyed on the
new signature.
"""

from __future__ import annotations

import streamlit as st

from auto_rag.ui.session import get_config, update_config

__all__ = ["render_config_page"]

_PROVIDERS: tuple[str, ...] = ("ollama", "openai")
_APPLIED_KEY = "ui.config_applied"


def render_config_page() -> None:
    """Render the configuration form and persist any changes."""
    config = get_config()
    st.subheader("LLM configuration")

    provider = st.selectbox(
        "Provider",
        list(_PROVIDERS),
        index=_PROVIDERS.index(config["provider"]) if config["provider"] in _PROVIDERS else 0,
        key="ui_config_provider",
    )
    model = st.text_input(
        "Model",
        value=str(config["model"]),
        placeholder="e.g. llama3.2",
        key="ui_config_model",
    )
    base_url = st.text_input(
        "Base URL",
        value=str(config["base_url"]),
        placeholder="http://localhost:11434",
        key="ui_config_base_url",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(config["temperature"]),
        step=0.05,
        key="ui_config_temperature",
    )

    st.caption(
        "Overrides apply to this session only. Leave a field empty to use the "
        "value from your configuration file."
    )
    if st.button("Apply configuration", type="primary", key="ui_config_apply"):
        update_config(
            provider=provider,
            model=model.strip(),
            base_url=base_url.strip(),
            temperature=float(temperature),
        )
        st.session_state[_APPLIED_KEY] = True
        st.rerun()

    if st.session_state.get(_APPLIED_KEY):
        st.success("Configuration applied to the current session.")
