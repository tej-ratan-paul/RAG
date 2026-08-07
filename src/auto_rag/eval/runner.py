"""Scoring a retriever against a labeled eval set.

For each example the retriever fetches a candidate pool (``pool_k``) and the
top ``k`` window is evaluated. ``recall@k`` is measured against the relevant
results found within the pool, so no exhaustive corpus relevance judgment is
required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from auto_rag.eval.loader import EvalExample
from auto_rag.eval.metrics import evaluate_ranking, mean_metric, metric_labels
from auto_rag.retrieval.models import RetrievedChunk

__all__ = ["EvaluationReport", "QueryEvaluation", "is_relevant", "run_evaluation"]


@dataclass(frozen=True)
class QueryEvaluation:
    """Retrieval metrics for one eval query."""

    query: str
    metrics: dict[str, float]
    elapsed_ms: float
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "metrics": self.metrics,
            "elapsed_ms": self.elapsed_ms,
            "retrieved_ids": self.retrieved_ids,
            "retrieved_sources": self.retrieved_sources,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated evaluation results across an eval set."""

    metrics: dict[str, float]
    queries: list[QueryEvaluation]
    top_k: int
    pool_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "pool_k": self.pool_k,
            "metrics": self.metrics,
            "queries": [query.to_dict() for query in self.queries],
        }


def is_relevant(chunk: RetrievedChunk, example: EvalExample) -> bool:
    """True when ``chunk`` matches the example's source or id labels."""
    if chunk.id in example.relevant_chunk_ids:
        return True
    source = (chunk.source or "").replace("\\", "/")
    basename = source.rsplit("/", 1)[-1]
    return basename in example.relevant_sources


def _relevance_flags(chunks: list[RetrievedChunk], example: EvalExample) -> list[bool]:
    return [is_relevant(chunk, example) for chunk in chunks]


def run_evaluation(
    retriever: Any,
    examples: list[EvalExample],
    top_k: int = 5,
    pool_k: int | None = None,
) -> EvaluationReport:
    """Evaluate ``retriever`` over ``examples`` and return the report."""
    pool_k = pool_k or max(top_k * 5, 20)
    queries: list[QueryEvaluation] = []
    for example in examples:
        started = time.perf_counter()
        pool = retriever.retrieve(example.query, top_k=pool_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        flags = _relevance_flags(pool, example)
        total_relevant = sum(flags)
        window = pool[:top_k]
        metrics = evaluate_ranking(
            flags[:top_k], top_k, total_relevant=total_relevant
        )
        queries.append(
            QueryEvaluation(
                query=example.query,
                metrics=metrics,
                elapsed_ms=elapsed_ms,
                retrieved_ids=[chunk.id for chunk in window],
                retrieved_sources=[chunk.source for chunk in window],
            )
        )
    metrics = {
        metric: mean_metric([query.metrics for query in queries], metric)
        for metric in metric_labels()
    }
    return EvaluationReport(
        metrics=metrics, queries=queries, top_k=top_k, pool_k=pool_k
    )
