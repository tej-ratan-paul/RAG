"""Tests for text cleaning."""

from __future__ import annotations

from auto_rag.ingestion.cleaning import clean_text


def test_rejoins_hyphenated_line_breaks() -> None:
    raw = "This is a manu-\nfacturing specification."
    assert clean_text(raw) == "This is a manufacturing specification."


def test_removes_soft_hyphens() -> None:
    raw = "A long\u00adword should stay together."
    assert "long\u00adword" not in clean_text(raw)


def test_collapses_whitespace() -> None:
    raw = "Many\t  spaces   here.\n\n\n\nToo many blank lines."
    cleaned = clean_text(raw)
    assert "Many spaces here." in cleaned
    assert "\n\n\n\n" not in cleaned


def test_replaces_non_breaking_spaces() -> None:
    raw = "Brake\u00a0pad\u00a0replacement"
    assert clean_text(raw) == "Brake pad replacement"


def test_removes_page_number_lines() -> None:
    raw = "Torque spec section.\n\n- 42 -\n\nNext section."
    cleaned = clean_text(raw)
    assert "- 42 -" not in cleaned
    assert "Torque spec section." in cleaned
    assert "Next section." in cleaned


def test_strips_surrounding_whitespace() -> None:
    assert clean_text("  \n  Hello world.\n  ") == "Hello world."


def test_empty_input() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""
