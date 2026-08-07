"""Ranking metrics for retrieval evaluation.

All functions operate on ``relevance`` — a sequence of booleans marking each
retrieved result (in rank order) as relevant. This keeps the metrics free of
any corpus assumptions; the runner builds the flags from the eval set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "evaluate_ranking",
    "hit_at_k",
    "mean_metric",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]

_METRIC_LABELS: tuple[str, ...] = ("hit@k", "precision@k", "recall@k", "mrr@k", "ndcg@k")


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 when at least one relevant result appears in the top ``k``."""
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    window = relevance[:k]
    if not window:
        return 0.0
    return sum(window) / len(window)


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """Fraction of relevant results (in the whole pool) found in the top ``k``."""
    if total_relevant <= 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def reciprocal_rank_at_k(relevance: Sequence[bool], k: int) -> float:
    """1/rank of the first relevant result, or 0.0 when none found in top ``k``."""
    for position, relevant in enumerate(relevance[:k], start=1):
        if relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevance: Sequence[bool], k: int) -> float:
    """Binary-relevance nDCG@k (1.0 for a perfect ranking)."""
    window = list(relevance[:k])
    if not any(window):
        return 0.0
    dcg = sum(1.0 / math.log2(index + 1) for index, rel in enumerate(window, start=1) if rel)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, sum(window) + 1))
    return dcg / ideal


def evaluate_ranking(
    relevance: Sequence[bool], k: int, total_relevant: int | None = None
) -> dict[str, float]:
    """Return the full metric set for a single query's ranking.

    Args:
        relevance: Per-result relevance flags in rank order.
        k: The evaluation cutoff.
        total_relevant: Number of relevant results in the retrieval pool;
            used for recall@k. Defaults to the relevant count within the
            top ``k`` window.
    """
    if total_relevant is None:
        total_relevant = sum(relevance[:k])
    return {
        "hit@k": hit_at_k(relevance, k),
        "precision@k": precision_at_k(relevance, k),
        "recall@k": recall_at_k(relevance, k, total_relevant),
        "mrr@k": reciprocal_rank_at_k(relevance, k),
        "ndcg@k": ndcg_at_k(relevance, k),
    }


def mean_metric(records: Sequence[dict[str, float]], metric: str) -> float:
    """Mean of ``metric`` across a sequence of per-query metric dicts."""
    if not records:
        return 0.0
    return sum(record.get(metric, 0.0) for record in records) / len(records)


def metric_labels() -> tuple[str, ...]:
    """Return the canonical metric names in report order."""
    return _METRIC_LABELS
