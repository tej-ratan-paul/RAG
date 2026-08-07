"""``auto-rag-bench``: measure retrieval and end-to-end RAG latency.

Latency samples are taken over a set of queries (optionally repeated), then
summarised with min / mean / p50 / p95 / p99 / max. Retrieval reports both the
total pipeline latency and each pipeline step (dense, lexical, fusion, MMR,
rerank) via :attr:`Retriever.last_timings`. Use ``--with-llm`` to also time
full grounded answers; benchmark conversations are deleted afterwards.

Usage::

    auto-rag-bench                          # retrieval latency, demo queries
    auto-rag-bench --queries queries.txt --repeats 5
    auto-rag-bench --with-llm --json-out bench.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from auto_rag.config import get_settings
from auto_rag.db.connection import Database
from auto_rag.db.repositories import ConversationRepository
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.llm import build_llm
from auto_rag.logging_config import get_logger, log_with_fields, setup_logging
from auto_rag.ops.stats import summarize
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.service import RAGService
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.reranker import build_reranker
from auto_rag.retrieval.retriever import Retriever

logger = get_logger(__name__)

DEFAULT_QUERIES: tuple[str, ...] = (
    "What is the recommended coolant mixture for the 2018 Camry?",
    "What torque should the caliper slide bolts be tightened to?",
    "When should the engine oil and oil filter be replaced?",
    "What are the possible causes of a P0300 misfire?",
    "How do I bleed the cooling system?",
    "What is the minimum front brake pad thickness?",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-bench",
        description="Benchmark AutoRAG retrieval and RAG response latency.",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="File with one query per line (blank lines and #-comments ignored).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to time each query (default: 3).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also time end-to-end RAG answers (requires a reachable LLM).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write the JSON report to this path.",
    )
    return parser


def load_queries(path: str | None, fallback: tuple[str, ...]) -> list[str]:
    """Read queries from ``path`` or fall back to the built-in set."""
    if not path:
        return list(fallback)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    queries = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if not queries:
        raise SystemExit(f"no queries found in {path}")
    return queries


def build_retriever(settings) -> Retriever:
    """Construct the full retrieval stack from settings."""
    store = build_vector_store(settings)
    bm25 = BM25Index(store.get_all_chunks(limit=100_000))
    reranker = build_reranker(
        enabled=settings.retrieval.rerank,
        model_name=settings.retrieval.reranker_model,
        device=settings.embeddings.device,
    )
    return Retriever(
        vector_store=store,
        embedding_provider=store.provider,
        config=settings.retrieval,
        bm25_index=bm25,
        reranker=reranker,
    )


def build_service(
    settings, retriever: Retriever
) -> tuple[RAGService, ConversationRepository, Database]:
    """Construct the RAG service (LLM + memory) with the given retriever."""
    db = Database.from_settings(settings)
    db.initialize()
    repo = ConversationRepository(db)
    llm = build_llm(settings.llm)
    service = RAGService(retriever, llm, ConversationMemory(repo))
    return service, repo, db


def benchmark_retrieval(
    retriever: Retriever, queries: list[str], repeats: int
) -> dict[str, dict[str, float]]:
    """Time ``retriever.retrieve`` per query and aggregate per-step timings."""
    total: list[float] = []
    steps: dict[str, list[float]] = {}
    for query in queries:
        for _ in range(repeats):
            started = time.perf_counter()
            retriever.retrieve(query)
            total.append((time.perf_counter() - started) * 1000)
            for step, elapsed in retriever.last_timings.items():
                steps.setdefault(step, []).append(elapsed)
    report: dict[str, dict[str, float]] = {"total_ms": summarize(total)}
    for step, samples in steps.items():
        report[f"{step}_ms"] = summarize(samples)
    return report


def benchmark_rag(
    service: RAGService,
    repo: ConversationRepository,
    queries: list[str],
    repeats: int,
) -> dict[str, float]:
    """Time full ``service.ask`` responses; delete the bench conversation."""
    conversation_id: int | None = None
    samples: list[float] = []
    try:
        for query in queries:
            for _ in range(repeats):
                started = time.perf_counter()
                service.ask(query, conversation_id=conversation_id)
                samples.append((time.perf_counter() - started) * 1000)
                if service.last_result is not None:
                    conversation_id = service.last_result.conversation_id
    finally:
        if conversation_id is not None:
            repo.delete(conversation_id)
    return summarize(samples)


def _print_report(report: dict[str, dict[str, float]]) -> None:
    print(f"{'metric':<16} {'count':>5} {'min':>9} {'mean':>9} {'p50':>9} "
          f"{'p95':>9} {'p99':>9} {'max':>9}")
    for key, stats in report.items():
        print(
            f"{key:<16} {int(stats['count']):>5} {stats['min']:>9.2f} "
            f"{stats['mean']:>9.2f} {stats['p50']:>9.2f} {stats['p95']:>9.2f} "
            f"{stats['p99']:>9.2f} {stats['max']:>9.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    queries = load_queries(args.queries, DEFAULT_QUERIES)
    logger.info("Benchmarking %d queries x %d repeats", len(queries), args.repeats)

    retriever = build_retriever(settings)
    retriever.retrieve(queries[0], top_k=1)  # warm-up: model + index load

    report: dict[str, dict[str, float]] = {}
    retrieval = benchmark_retrieval(retriever, queries, args.repeats)
    report["retrieval_total_ms"] = retrieval["total_ms"]
    for step, stats in retrieval.items():
        if step != "total_ms":
            report[f"retrieval_{step}"] = stats

    if args.with_llm:
        service, repo, db = build_service(settings, retriever)
        try:
            report["rag_response_ms"] = benchmark_rag(
                service, repo, queries, args.repeats
            )
        finally:
            db.close()

    _print_report(report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report written to %s", args.json_out)

    log_with_fields(
        logger,
        logging.INFO,
        "benchmark complete",
        **{k: v["p50"] for k, v in report.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
