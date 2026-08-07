"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from auto_rag.eval.metrics import (
    evaluate_ranking,
    hit_at_k,
    mean_metric,
    metric_labels,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hit_at_k() -> None:
    assert hit_at_k([False, True, False], 2) == 1.0
    assert hit_at_k([False, False], 2) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k([True, False, True], 2) == 0.5
    assert precision_at_k([True, False, True], 3) == pytest.approx(2 / 3)
    assert precision_at_k([], 2) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k([True, False], 2, total_relevant=4) == 0.25
    assert recall_at_k([False, False, False], 3, total_relevant=3) == 0.0
    assert recall_at_k([True, True], 2, total_relevant=0) == 0.0


def test_reciprocal_rank_at_k() -> None:
    assert reciprocal_rank_at_k([False, True, False], 3) == 0.5
    assert reciprocal_rank_at_k([True, False], 2) == 1.0
    assert reciprocal_rank_at_k([False, False], 2) == 0.0


def test_ndcg_perfect_ranking() -> None:
    assert ndcg_at_k([True, True], 2) == 1.0
    assert ndcg_at_k([True, True, True], 3) == 1.0


def test_ndcg_penalises_imperfect_ranking() -> None:
    perfect = ndcg_at_k([True, True, True], 3)
    imperfect = ndcg_at_k([True, False, True], 3)
    assert 0.0 < imperfect < perfect
    assert ndcg_at_k([False, False], 2) == 0.0


def test_evaluate_ranking_shape() -> None:
    metrics = evaluate_ranking([True, False, True], 3, total_relevant=5)
    assert set(metrics) == {"hit@k", "precision@k", "recall@k", "mrr@k", "ndcg@k"}
    assert metrics["hit@k"] == 1.0
    assert metrics["mrr@k"] == 1.0


def test_mean_metric() -> None:
    records = [{"a": 1.0}, {"a": 2.0}]
    assert mean_metric(records, "a") == 1.5
    assert mean_metric([], "a") == 0.0


def test_metric_labels_are_stable() -> None:
    assert metric_labels() == ("hit@k", "precision@k", "recall@k", "mrr@k", "ndcg@k")
