"""Command-line interface for document ingestion.

Usage::

    python -m auto_rag.ingestion.cli --directory data/documents
    python -m auto_rag.ingestion.cli --file manual.pdf --doc-type service_manual
    python -m auto_rag.ingestion.cli --csv parts.csv
    python -m auto_rag.ingestion.cli --sqlite workshop.db --table parts
    python -m auto_rag.ingestion.cli --sqlite workshop.db --query "SELECT * FROM dtc_codes WHERE severity='high'"
    python -m auto_rag.ingestion.cli --sql-url "postgresql://user:pass@host/db" --table vehicles
    python -m auto_rag.ingestion.cli --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_rag.config import get_settings
from auto_rag.db.connection import Database
from auto_rag.db.seeder import seed_demo_data
from auto_rag.ingestion.chunking import Chunker
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.ingestion.metadata import MetadataExtractor
from auto_rag.ingestion.pipeline import IngestionPipeline
from auto_rag.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

_DOC_TYPE_CHOICES = [
    "service_manual",
    "repair_manual",
    "dtc",
    "tsb",
    "wiring_diagram",
    "tabular",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-ingest",
        description="Ingest documents into the AutoRAG vector store.",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--directory", type=Path, help="Directory of documents to ingest.")
    source.add_argument("--file", type=Path, help="Single document to ingest.")
    source.add_argument("--csv", type=Path, help="Single CSV file to ingest (row per chunk).")
    sql = parser.add_mutually_exclusive_group(required=False)
    sql.add_argument("--sqlite", type=Path, help="SQLite database file to ingest rows from.")
    sql.add_argument(
        "--sql-url",
        type=str,
        help=(
            "Remote SQL connection URL (postgresql://... or mysql://...). "
            "Requires the matching optional driver package."
        ),
    )
    parser.add_argument("--table", default=None, help="Table to ingest (SELECT * FROM <table>).")
    parser.add_argument(
        "--query", default=None, help="Raw SQL query to ingest instead of a whole table."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of SQL rows to ingest.",
    )
    parser.add_argument(
        "--doc-type",
        choices=_DOC_TYPE_CHOICES,
        default=None,
        help="Explicit document type override.",
    )
    parser.add_argument("--force", action="store_true", help="Re-index already indexed files.")
    parser.add_argument("--reset", action="store_true", help="Delete all indexed chunks first.")
    parser.add_argument("--seed", action="store_true", help="Seed demo data into SQLite.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_source_args(args)

    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    database = Database.from_settings(settings)
    database.initialize()

    if args.seed:
        seed_demo_data(database)
        logger.info("Demo data seeded into %s", database.path)

    vector_store = build_vector_store(settings)
    if args.reset:
        vector_store.reset()
        logger.info("Vector collection reset (%s chunks removed)", vector_store.count())

    pipeline = IngestionPipeline(
        database=database,
        vector_store=vector_store,
        chunker=Chunker(
            size=settings.chunking.size,
            overlap=settings.chunking.overlap,
        ),
        extractor=MetadataExtractor(),
    )

    results: list = []
    if args.sqlite or args.sql_url:
        source = args.sqlite or args.sql_url
        results.append(
            pipeline.ingest_sql(
                source,
                table=args.table,
                query=args.query,
                limit=args.limit,
                force=args.force,
            )
        )
    elif args.file:
        results.append(pipeline.ingest_path(args.file, doc_type=args.doc_type, force=args.force))
    elif args.csv:
        results.append(pipeline.ingest_csv(args.csv, doc_type=args.doc_type, force=args.force))
    elif args.directory:
        results = pipeline.ingest_directory(args.directory, doc_type=args.doc_type, force=args.force)
    else:
        build_parser().error("One of --directory, --file, --csv, --sqlite, or --sql-url is required")
        return 2

    _report(results, vector_store.count())
    return 1 if any(r.is_error for r in results) else 0


def _validate_source_args(args: argparse.Namespace) -> None:
    """Reject invalid source flag combinations before any heavy setup."""
    parser = build_parser()
    has_file_source = bool(args.directory or args.file or args.csv)
    has_sql_source = bool(args.sqlite or args.sql_url)
    if has_file_source and has_sql_source:
        parser.error("--sqlite/--sql-url cannot be combined with --directory/--file/--csv")
    if not has_file_source and not has_sql_source:
        parser.error("One of --directory, --file, --csv, --sqlite, or --sql-url is required")
    if not has_sql_source and (args.table or args.query or args.limit is not None):
        parser.error("--table/--query/--limit require --sqlite or --sql-url")


def _report(results, total_chunks: int) -> None:
    indexed = sum(1 for r in results if r.status == "indexed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info(
        "Ingestion complete: indexed=%d skipped=%d failed=%d total_chunks_in_store=%d",
        indexed,
        skipped,
        failed,
        total_chunks,
    )
    for result in results:
        if result.is_error:
            logger.error("  FAILED %s: %s", result.path, result.error)
        else:
            logger.info(
                "  %-8s %s (%d chunks)",
                result.status.upper(),
                result.path,
                result.chunk_count,
            )


if __name__ == "__main__":
    sys.exit(main())
