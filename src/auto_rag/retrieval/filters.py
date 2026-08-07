"""Metadata filtering for retrieval.

Converts a :class:`RetrievalFilter` into a Chroma ``where`` clause for the
dense channel and into a plain predicate for the lexical (BM25) channel so
both paths honour the same narrowing criteria.
"""

from __future__ import annotations

from typing import Any

from auto_rag.retrieval.models import RetrievalFilter

__all__ = ["to_where", "matches_filter", "matches_document"]

_COMPARATORS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte")


def _normalise(value: Any) -> Any:
    """Treat empty strings as missing and compare text case-insensitively."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.lower() if stripped else None
    return value


def to_where(retrieval_filter: RetrievalFilter) -> dict[str, Any]:
    """Build a Chroma ``where`` clause from a :class:`RetrievalFilter`.

    Returns an empty dict when no criteria are set (callers omit the filter).
    """
    conditions: list[dict[str, Any]] = []
    for field, value in retrieval_filter.as_dict().items():
        if isinstance(value, str):
            conditions.append({field: {"$eq": value.strip()}})
        else:
            conditions.append({field: value})
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def matches_filter(retrieval_filter: RetrievalFilter, metadata: dict[str, Any]) -> bool:
    """Predicate form of :class:`RetrievalFilter` for lexically-scored chunks."""
    for field, expected in retrieval_filter.as_dict().items():
        stored = _normalise(metadata.get(field))
        if stored != _normalise(expected):
            return False
    return True


def matches_document(
    criteria: dict[str, Any], metadata: dict[str, Any]
) -> bool:
    """Match a raw metadata dict (document record) against plain criteria.

    Supports simple equality values and ``$eq``/``$in``/``$ne`` operators so
    callers can filter document rows with the same semantics as Chroma.
    """
    for field, condition in criteria.items():
        if isinstance(condition, dict) and any(
            op in condition for op in _COMPARATORS
        ):
            operator = next(op for op in _COMPARATORS if op in condition)
            operand = condition[operator]
            stored = _normalise(metadata.get(field))
            expected = _normalise(operand)
            if operator == "$eq" and stored != expected:
                return False
            if operator == "$ne" and stored == expected:
                return False
            if operator == "$in" and stored not in _normalise_list(operand):
                return False
            if operator == "$gt" and not (stored is not None and stored > expected):
                return False
            if operator == "$gte" and not (stored is not None and stored >= expected):
                return False
            if operator == "$lt" and not (stored is not None and stored < expected):
                return False
            if operator == "$lte" and not (stored is not None and stored <= expected):
                return False
        elif _normalise(metadata.get(field)) != _normalise(condition):
            return False
    return True


def _normalise_list(values: Any) -> list[Any]:
    if not isinstance(values, list):
        values = [values]
    return [_normalise(value) for value in values]
