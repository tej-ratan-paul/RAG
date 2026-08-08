"""Ingestion pipeline.

Orchestrates the full load -> clean -> metadata -> chunk -> embed -> index
flow for files (PDF, text, CSV) and SQL sources (SQLite files or remote SQL
URLs), tracking every source in SQLite and deduplicating by content hash.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from auto_rag.constants import DOCUMENT_TYPE_SERVICE_MANUAL, DOCUMENT_TYPE_TABULAR
from auto_rag.db.connection import Database
from auto_rag.db.models import DocType, DocumentRecord
from auto_rag.db.repositories import DocumentRepository
from auto_rag.errors import IngestionError
from auto_rag.ingestion.chunking import Chunker
from auto_rag.ingestion.cleaning import clean_text
from auto_rag.ingestion.loaders import (
    PageContent,
    SQLLoader,
    loader_for,
    sql_source_label,
)
from auto_rag.ingestion.metadata import DocumentMetadata, MetadataExtractor
from auto_rag.ingestion.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting a single file."""

    path: str
    status: str  # "indexed" | "skipped" | "failed"
    document_id: int | None = None
    chunk_count: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status == "failed"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionPipeline:
    """Runs documents through the ingestion pipeline."""

    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        chunker: Chunker,
        extractor: MetadataExtractor | None = None,
    ) -> None:
        self.database = database
        self.vector_store = vector_store
        self.chunker = chunker
        self.extractor = extractor or MetadataExtractor()
        self.documents = DocumentRepository(database)

    # ------------------------------------------------------------------ #
    def ingest_path(
        self,
        path: Path,
        doc_type: DocType | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Ingest a single file.

        Args:
            path: PDF, text, or CSV file to ingest.
            doc_type: Optional explicit document type.
            force: Re-index even if the file was already indexed.

        Returns:
            An :class:`IngestResult` describing the outcome.
        """
        path = Path(path)
        file_hash = sha256_file(path)
        fallback_type = _fallback_type_for(path)

        existing = self.documents.get_by_hash(file_hash)
        if existing and existing.status == "indexed" and not force:
            logger.info("Skipping already indexed file %s", path)
            return IngestResult(
                path=str(path), status="skipped", document_id=existing.id
            )

        document_id: int | None = None
        try:
            if existing is None:
                record = DocumentRecord(
                    source_path=str(path),
                    file_hash=file_hash,
                    doc_type=doc_type or fallback_type or "service_manual",
                    title=path.stem,
                )
                document_id = self.documents.add(record)
            else:
                document_id = existing.id
                self.documents.update_status(document_id, "pending")
            return self._process(path, doc_type, document_id, fallback_type)
        except IngestionError as exc:
            logger.exception("Ingestion failed for %s", path)
            if document_id is not None:
                self.documents.update_status(document_id, "failed", error=str(exc))
            return IngestResult(
                path=str(path), status="failed", document_id=document_id, error=str(exc)
            )

    def ingest_csv(
        self,
        path: Path,
        doc_type: DocType | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Ingest a single CSV file (rows become individual chunks)."""
        path = Path(path)
        if path.suffix.lower() != ".csv":
            raise IngestionError(f"Not a CSV file: {path}")
        return self.ingest_path(path, doc_type=doc_type, force=force)

    def ingest_sql(
        self,
        source: str | Path,
        *,
        table: str | None = None,
        query: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Ingest rows from a SQL source as a one-shot snapshot.

        Args:
            source: A SQLite database file path or a remote SQL connection URL
                (``postgresql://...``, ``mysql://...``). Credentials in the
                source label are redacted.
            table: Table to read (``SELECT * FROM <table>``).
            query: Raw SQL query. If both are given, ``query`` wins.
            limit: Optional cap on the number of rows ingested.
            force: Re-index even if this source was already ingested.

        Returns:
            An :class:`IngestResult` describing the outcome.
        """
        source_str = str(source)
        loader = SQLLoader(table=table, query=query, limit=limit)
        pages = loader.load_source(source_str)
        if not pages:
            raise IngestionError(f"No rows returned from SQL source {source_str}")

        fingerprint = _sql_fingerprint(source_str, table, query, pages)
        label = sql_source_label(source_str, table=table, query=query)
        title = table or "sql source"
        label_path = Path(title)

        existing = self.documents.get_by_hash(fingerprint)
        if existing and existing.status == "indexed" and not force:
            logger.info("Skipping already indexed SQL source %s", label)
            return IngestResult(
                path=label, status="skipped", document_id=existing.id
            )

        document_id: int | None = None
        try:
            if existing is None:
                record = DocumentRecord(
                    source_path=label,
                    file_hash=fingerprint,
                    doc_type="tabular",
                    title=title,
                )
                document_id = self.documents.add(record)
            else:
                document_id = existing.id
                self.documents.update_status(document_id, "pending")
            return self._index_pages(
                pages=pages,
                label_path=label_path,
                source=label,
                doc_type=None,
                fallback_type=DOCUMENT_TYPE_TABULAR,
                document_id=document_id,
                display_path=label,
            )
        except IngestionError as exc:
            logger.exception("Ingestion failed for SQL source %s", label)
            if document_id is not None:
                self.documents.update_status(document_id, "failed", error=str(exc))
            return IngestResult(
                path=label, status="failed", document_id=document_id, error=str(exc)
            )

    def ingest_directory(
        self,
        directory: Path,
        doc_type: DocType | None = None,
        force: bool = False,
    ) -> list[IngestResult]:
        """Ingest every supported file under ``directory`` (recursively)."""
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        results: list[IngestResult] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            try:
                loader_for(path)
            except IngestionError:
                continue
            results.append(self.ingest_path(path, doc_type=doc_type, force=force))
        return results

    # ------------------------------------------------------------------ #
    def _process(
        self,
        path: Path,
        doc_type: DocType | None,
        document_id: int,
        fallback_type: DocType | None = None,
    ) -> IngestResult:
        loader = loader_for(path)
        pages = loader.load(path)
        return self._index_pages(
            pages=pages,
            label_path=path,
            source=str(path),
            doc_type=doc_type,
            fallback_type=fallback_type,
            document_id=document_id,
            display_path=str(path),
        )

    def _index_pages(
        self,
        *,
        pages: list[PageContent],
        label_path: Path,
        source: str,
        doc_type: DocType | None,
        fallback_type: DocType | None,
        document_id: int,
        display_path: str,
    ) -> IngestResult:
        """Clean, chunk, embed, and index already-loaded pages."""
        if not pages:
            raise IngestionError(f"No extractable content in {display_path}")

        cleaned = _clean_pages(pages)
        metadata: DocumentMetadata = self.extractor.extract(
            label_path,
            cleaned,
            doc_type,
            fallback_type=fallback_type or DOCUMENT_TYPE_SERVICE_MANUAL,
        )
        chunks = self.chunker.chunk_pages(cleaned, metadata.to_dict())

        self.vector_store.add_chunks(chunks, source=source)
        self.documents.update_metadata(
            document_id,
            doc_type=metadata.doc_type,
            title=metadata.title,
            make=metadata.make,
            model=metadata.model,
            year=metadata.year,
            engine=metadata.engine,
            vin=metadata.vin,
            page_count=len(cleaned),
        )
        self.documents.update_status(
            document_id,
            "indexed",
            chunk_count=len(chunks),
        )
        logger.info(
            "Indexed %s: %d chunks, %d rows/pages (type=%s)",
            display_path,
            len(chunks),
            len(cleaned),
            metadata.doc_type,
        )
        return IngestResult(
            path=display_path,
            status="indexed",
            document_id=document_id,
            chunk_count=len(chunks),
            metadata=metadata.to_dict(),
        )


def _fallback_type_for(path: Path) -> DocType | None:
    """Return the default doc type for a file's suffix (or None)."""
    if path.suffix.lower() == ".csv":
        return DOCUMENT_TYPE_TABULAR
    return None


def _sql_fingerprint(
    source: str,
    table: str | None,
    query: str | None,
    pages: list[PageContent],
) -> str:
    """Content hash for a SQL snapshot: query identity + every row's text."""
    statement = query or table or ""
    payload = f"{source}|{statement}\n" + "\n".join(page.text for page in pages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_pages(pages: list[PageContent]) -> list[PageContent]:
    """Clean page text and drop pages with no remaining content."""
    cleaned: list[PageContent] = []
    for page in pages:
        text = clean_text(page.text)
        if text:
            cleaned.append(PageContent(page_number=page.page_number, text=text, metadata=page.metadata))
    return cleaned
