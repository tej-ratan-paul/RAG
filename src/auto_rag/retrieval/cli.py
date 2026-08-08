"""Retrieval command-line interface.

Usage::

    python -m auto_rag.retrieval.cli --query "brake pad thickness"
    python -m auto_rag.retrieval.cli --query "misfire" --make Toyota --top-k 5
    python -m auto_rag.retrieval.cli --query "P0300" --doc-type dtc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_rag.config import get_settings
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.ingestion.embeddings import get_embedding_provider
from auto_rag.logging_config import get_logger, setup_logging
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.models import RetrievalFilter
from auto_rag.retrieval.reranker import build_reranker
from auto_rag.retrieval.retriever import Retriever

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-retrieve",
        description="Query the AutoRAG vector store.",
    )
    parser.add_argument("--query", required=True, help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results.")
    parser.add_argument("--make", default=None, help="Filter by vehicle make.")
    parser.add_argument("--model", default=None, help="Filter by vehicle model.")
    parser.add_argument("--year", type=int, default=None, help="Filter by model year.")
    parser.add_argument(
        "--doc-type",
        choices=["service_manual", "repair_manual", "dtc", "tsb", "wiring_diagram", "tabular"],
        default=None,
        help="Filter by document type.",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    if args.no_rerank:
        settings.retrieval.rerank = False

    store = build_vector_store(settings)
    provider = get_embedding_provider(
        settings.embeddings, cache_path=settings.embedding_cache_path
    )
    chunks = store.get_all_chunks(limit=100_000)
    bm25_index = BM25Index(chunks)
    reranker = build_reranker(
        enabled=settings.retrieval.rerank,
        model_name=settings.retrieval.reranker_model,
        device=settings.embeddings.device,
    )
    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        config=settings.retrieval,
        bm25_index=bm25_index,
        reranker=reranker,
    )

    retrieval_filter = RetrievalFilter(
        make=args.make,
        model=args.model,
        year=args.year,
        doc_type=args.doc_type,
    )
    results = retriever.retrieve(
        args.query, top_k=args.top_k, retrieval_filter=retrieval_filter
    )

    if not results:
        logger.info("No results found for query %r", args.query)
        return 1

    for index, chunk in enumerate(results, start=1):
        source = Path(chunk.source).name if chunk.source else "?"
        page = chunk.page or "?"
        logger.info(
            "%2d. [%.3f] %s (p%s, %s/%s) %s",
            index,
            chunk.score,
            source,
            page,
            chunk.make or "?",
            chunk.model or "?",
            chunk.text[:160].replace("\n", " "),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
