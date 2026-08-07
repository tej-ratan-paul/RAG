"""Ingestion pipeline.

Orchestrates the full load -> clean -> metadata -> chunk -> embed -> index
flow for a file or directory, tracking every document in SQLite and
deduplicating by content hash.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from auto_rag.db.connection import Database
from auto_rag.db.models import DocType, DocumentRecord
from auto_rag.db.repositories import DocumentRepository
from auto_rag.errors import IngestionError
from auto_rag.ingestion.chunking import Chunker
from auto_rag.ingestion.cleaning import clean_text
from auto_rag.ingestion.loaders import PageContent, loader_for
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
            path: PDF or text file to ingest.
            doc_type: Optional explicit document type.
            force: Re-index even if the file was already indexed.

        Returns:
            An :class:`IngestResult` describing the outcome.
        """
        path = Path(path)
        file_hash = sha256_file(path)

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
                    doc_type=doc_type or "service_manual",
                    title=path.stem,
                )
                document_id = self.documents.add(record)
            else:
                document_id = existing.id
                self.documents.update_status(document_id, "pending")
            return self._process(path, doc_type, document_id)
        except IngestionError as exc:
            logger.exception("Ingestion failed for %s", path)
            if document_id is not None:
                self.documents.update_status(document_id, "failed", error=str(exc))
            return IngestResult(
                path=str(path), status="failed", document_id=document_id, error=str(exc)
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
    def _process(self, path: Path, doc_type: DocType | None, document_id: int) -> IngestResult:
        loader = loader_for(path)
        pages = loader.load(path)
        if not pages:
            raise IngestionError(f"No extractable content in {path}")

        cleaned = _clean_pages(pages)
        metadata: DocumentMetadata = self.extractor.extract(path, cleaned, doc_type)
        chunks = self.chunker.chunk_pages(cleaned, metadata.to_dict())

        self.vector_store.add_chunks(chunks, source=str(path))
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
            "Indexed %s: %d chunks, %d pages (type=%s)",
            path.name,
            len(chunks),
            len(cleaned),
            metadata.doc_type,
        )
        return IngestResult(
            path=str(path),
            status="indexed",
            document_id=document_id,
            chunk_count=len(chunks),
            metadata=metadata.to_dict(),
        )


def _clean_pages(pages: list[PageContent]) -> list[PageContent]:
    """Clean page text and drop pages with no remaining content."""
    cleaned: list[PageContent] = []
    for page in pages:
        text = clean_text(page.text)
        if text:
            cleaned.append(PageContent(page_number=page.page_number, text=text, metadata=page.metadata))
    return cleaned
