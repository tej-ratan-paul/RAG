"""Eval-set loading and validation.

A labeled eval set is JSON::

    {
        "version": 1,
        "description": "Demo corpus",
        "queries": [
            {
                "query": "What torque for caliper slide bolts?",
                "relevant_sources": ["toyota_camry_2018_service_manual.pdf"],
                "relevant_chunk_ids": []
            }
        ]
    }

A chunk is relevant when its stored ``source`` basename matches a
``relevant_sources`` entry or its id matches a ``relevant_chunk_ids`` entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_rag.errors import ConfigurationError

__all__ = ["EvalExample", "load_eval_set"]


@dataclass(frozen=True)
class EvalExample:
    """One labeled query with its ground-truth relevant chunks."""

    query: str
    relevant_sources: tuple[str, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()

    @property
    def has_labels(self) -> bool:
        return bool(self.relevant_sources or self.relevant_chunk_ids)


def _extract_entries(data: Any) -> list[Any]:
    if isinstance(data, dict):
        entries = data.get("queries")
        if entries is None:
            raise ConfigurationError(
                "Eval set must contain a 'queries' list"
            )
    elif isinstance(data, list):
        entries = data
    else:
        raise ConfigurationError(
            "Eval set must be a JSON object with a 'queries' list or a bare list"
        )
    if not isinstance(entries, list):
        raise ConfigurationError("'queries' must be a JSON array")
    return entries


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """Load and validate an eval-set file, raising ConfigurationError on issues."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Could not read eval set {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in eval set {path}: {exc}") from exc

    examples: list[EvalExample] = []
    for index, entry in enumerate(_extract_entries(data)):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"Eval set {path}: entry {index} must be an object")
        query = str(entry.get("query", "")).strip()
        if not query:
            raise ConfigurationError(f"Eval set {path}: entry {index} needs a non-empty 'query'")
        sources = tuple(str(source) for source in entry.get("relevant_sources", []))
        ids = tuple(str(chunk_id) for chunk_id in entry.get("relevant_chunk_ids", []))
        if not sources and not ids:
            raise ConfigurationError(
                f"Eval set {path}: entry {index} needs 'relevant_sources' "
                "or 'relevant_chunk_ids'"
            )
        examples.append(
            EvalExample(query=query, relevant_sources=sources, relevant_chunk_ids=ids)
        )
    if not examples:
        raise ConfigurationError(f"Eval set {path} contains no queries")
    return examples
