"""RAG chat command-line interface.

Usage::

    python -m auto_rag.rag.cli --query "How do I replace front brake pads?"
    python -m auto_rag.rag.cli --query "P0300 diagnosis" --make Toyota --doc-type dtc
    python -m auto_rag.rag.cli --query "torque spec" --conversation-id 3 --stream
"""

from __future__ import annotations

import argparse
import sys

from auto_rag.config import get_settings
from auto_rag.constants import DOCUMENT_TYPES
from auto_rag.db.connection import Database
from auto_rag.db.repositories import ConversationRepository
from auto_rag.errors import AutoRAGError
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.llm import build_llm
from auto_rag.logging_config import get_logger, setup_logging
from auto_rag.rag.citations import format_citations
from auto_rag.rag.memory import ConversationMemory
from auto_rag.rag.models import RAGResult
from auto_rag.rag.service import RAGService
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.models import RetrievalFilter
from auto_rag.retrieval.reranker import build_reranker
from auto_rag.retrieval.retriever import Retriever

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-ask",
        description="Ask the AutoRAG repair assistant a grounded question.",
    )
    parser.add_argument("--query", required=True, help="Your repair question.")
    parser.add_argument("--make", default=None, help="Vehicle make filter.")
    parser.add_argument("--model", default=None, help="Vehicle model filter.")
    parser.add_argument("--year", type=int, default=None, help="Model year filter.")
    parser.add_argument(
        "--doc-type",
        choices=list(DOCUMENT_TYPES),
        default=None,
        help="Document type filter.",
    )
    parser.add_argument(
        "--conversation-id",
        type=int,
        default=None,
        help="Reuse an existing conversation for context.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Number of retrieval results to use."
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream the answer as it is generated."
    )
    parser.add_argument(
        "--provider", default=None, help="LLM provider override (ollama|openai)."
    )
    parser.add_argument(
        "--model-name", dest="model_name", default=None, help="LLM model override."
    )
    parser.add_argument(
        "--base-url", default=None, help="LLM API base URL override."
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable cross-encoder reranking."
    )
    return parser


def _print_meta(result: RAGResult) -> None:
    if result.sources:
        print("Sources:")
        print(format_citations(result.sources))
    else:
        print("No sources found.")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence:.2f}")
    if result.safety_notes:
        print("Safety notes:")
        for note in result.safety_notes:
            print(f"  - {note}")
    if result.conversation_id is not None:
        print(f"Conversation: {result.conversation_id}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    if args.provider:
        settings.llm.provider = args.provider
    if args.model_name:
        settings.llm.model = args.model_name
    if args.base_url:
        settings.llm.base_url = args.base_url
    if args.no_rerank:
        settings.retrieval.rerank = False

    db = Database.from_settings(settings)
    db.initialize()
    try:
        store = build_vector_store(settings)
        bm25 = BM25Index(store.get_all_chunks(limit=100_000))
        reranker = build_reranker(
            enabled=settings.retrieval.rerank,
            model_name=settings.retrieval.reranker_model,
            device=settings.embeddings.device,
        )
        retriever = Retriever(
            vector_store=store,
            embedding_provider=store.provider,
            config=settings.retrieval,
            bm25_index=bm25,
            reranker=reranker,
        )
        llm = build_llm(settings.llm)
        memory = ConversationMemory(ConversationRepository(db))
        service = RAGService(retriever, llm, memory)

        retrieval_filter = RetrievalFilter(
            make=args.make,
            model=args.model,
            year=args.year,
            doc_type=args.doc_type,
        )

        try:
            if args.stream:
                print("AutoRAG: ", end="", flush=True)
                for piece in service.ask_stream(
                    args.query,
                    conversation_id=args.conversation_id,
                    retrieval_filter=retrieval_filter,
                    top_k=args.top_k,
                ):
                    print(piece, end="", flush=True)
                print("\n")
                result = service.last_result
            else:
                result = service.ask(
                    args.query,
                    conversation_id=args.conversation_id,
                    retrieval_filter=retrieval_filter,
                    top_k=args.top_k,
                )
                print(result.answer)
        except AutoRAGError as exc:
            logger.error("Request failed: %s", exc)
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if result is not None:
            _print_meta(result)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
