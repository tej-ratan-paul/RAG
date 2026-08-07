"""End-to-end ingestion pipeline tests (deterministic embeddings, offline)."""

from __future__ import annotations

from auto_rag.db.connection import Database
from auto_rag.db.repositories import DocumentRepository
from auto_rag.ingestion.chunking import Chunker
from auto_rag.ingestion.metadata import MetadataExtractor
from auto_rag.ingestion.pipeline import IngestionPipeline, sha256_file
from auto_rag.ingestion.vectorstore import VectorStore


def _pipeline(db: Database, store: VectorStore) -> IngestionPipeline:
    return IngestionPipeline(
        database=db,
        vector_store=store,
        chunker=Chunker(size=200, overlap=40),
        extractor=MetadataExtractor(source_hints=3),
    )


def test_ingest_pdf_end_to_end(db: Database, vector_store: VectorStore, make_pdf) -> None:
    pdf = make_pdf(
        [
            "2018 Toyota Camry engine service information.",
            "The camshaft position sensor is located at the front of the cylinder head.",
        ]
    )
    result = _pipeline(db, vector_store).ingest_path(pdf)
    assert result.status == "indexed"
    assert result.document_id is not None
    assert result.chunk_count >= 1

    record = DocumentRepository(db).get_by_id(result.document_id)
    assert record is not None
    assert record.status == "indexed"
    assert record.doc_type == "service_manual"
    assert record.make == "Toyota"
    assert record.model == "Camry"
    assert record.year == 2018
    assert vector_store.count() == result.chunk_count


def test_ingest_skips_already_indexed(db: Database, vector_store: VectorStore, make_pdf) -> None:
    pdf = make_pdf(["Duplicate content to ingest twice."])
    pipeline = _pipeline(db, vector_store)
    first = pipeline.ingest_path(pdf)
    second = pipeline.ingest_path(pdf)
    assert first.status == "indexed"
    assert second.status == "skipped"
    assert vector_store.count() == first.chunk_count


def test_force_reindexes(db: Database, vector_store: VectorStore, make_pdf) -> None:
    pdf = make_pdf(["Reindex me."])
    pipeline = _pipeline(db, vector_store)
    first = pipeline.ingest_path(pdf)
    second = pipeline.ingest_path(pdf, force=True)
    assert first.status == "indexed"
    assert second.status == "indexed"
    assert vector_store.count() == first.chunk_count  # dedup by chunk id


def test_ingest_directory_recurses(db: Database, vector_store: VectorStore, make_pdf, tmp_path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    make_pdf(["First document about oil filters."])
    make_pdf(["Second distinct document about brake rotors."], name="dtc_p0300_codes.pdf")
    sub / "brakes.pdf"
    make_pdf(["Third document about transmission fluid."], name=sub / "gearbox.pdf")

    results = _pipeline(db, vector_store).ingest_directory(tmp_path)
    assert len(results) == 3
    assert all(r.status == "indexed" for r in results)


def test_unsupported_file_reports_failed(db: Database, vector_store: VectorStore, tmp_path) -> None:
    bad = tmp_path / "archive.zip"
    bad.write_bytes(b"PK\x03\x04 not really")
    result = _pipeline(db, vector_store).ingest_path(bad)
    assert result.is_error
    record = DocumentRepository(db).get_by_hash(sha256_file(bad))
    assert record is None or record.status != "indexed"


def test_retrieval_roundtrip_after_ingestion(
    db: Database, vector_store: VectorStore, make_pdf
) -> None:
    pdf = make_pdf(
        [
            "2018 Toyota Camry front brake pad replacement procedure.",
            "Alternator replacement requires removing the drive belt.",
        ]
    )
    _pipeline(db, vector_store).ingest_path(pdf)

    hits = vector_store.similarity_search("front brake pad replacement procedure", top_k=2)
    assert hits
    assert "brake" in hits[0]["text"].lower()
    assert hits[0]["metadata"]["make"] == "Toyota"
