"""Tests for benchmark statistics helpers."""

from __future__ import annotations

import pytest

from auto_rag.ops.stats import percentile, summarize


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.25) == 1.75
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0


def test_percentile_handles_small_inputs() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.5) == 7.0


def test_summarize_aggregates() -> None:
    stats = summarize([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5
    assert stats["p50"] == 2.5
    assert pytest.approx(stats["p95"]) == 3.85


def test_summarize_empty() -> None:
    stats = summarize([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["p50"] == 0.0
