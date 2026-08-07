"""Datetime helpers used across the persistence layer."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")
