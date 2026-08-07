"""Small statistical helpers for benchmark reports.

Percentiles use linear interpolation between the two nearest order
statistics (matching ``numpy.percentile`` default behaviour) so results are
stable and dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["percentile", "summarize"]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` (0..1) of ascending data."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def summarize(
    values: Sequence[float], labels: Sequence[str] = ("p50", "p95", "p99")
) -> dict[str, float]:
    """Summarise ``values``: count, min, mean, max and requested percentiles."""
    ordered = sorted(values)
    stats: dict[str, float] = {
        "count": float(len(values)),
        "min": float(ordered[0]) if ordered else 0.0,
        "max": float(ordered[-1]) if ordered else 0.0,
        "mean": float(sum(values) / len(values)) if values else 0.0,
    }
    for label in labels:
        stats[str(label)] = percentile(ordered, float(label[1:]) / 100.0)
    return stats
