"""End-to-end ingestion pipeline tests (deterministic embeddings, offline)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from auto_rag.db.connection import Database
from auto_rag.db.repositories import DocumentRepository
from auto_rag.errors import IngestionError
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


# --------------------------------------------------------------------- #
# CSV ingestion
# --------------------------------------------------------------------- #
def _write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "parts.csv"
    path.write_text(
        "part_number,name,unit_price\n"
        "BP-101,Brake pads,45.50\n"
        "OF-202,Oil filter,8.25\n",
        encoding="utf-8",
    )
    return path


def test_ingest_csv_end_to_end(db: Database, vector_store: VectorStore, tmp_path: Path) -> None:
    result = _pipeline(db, vector_store).ingest_csv(_write_csv(tmp_path))
    assert result.status == "indexed"
    assert result.chunk_count == 2
    assert result.metadata["doc_type"] == "tabular"

    record = DocumentRepository(db).get_by_id(result.document_id)
    assert record is not None
    assert record.status == "indexed"
    assert record.doc_type == "tabular"
    assert record.page_count == 2
    assert vector_store.count() == 2


def test_ingest_csv_skips_on_repeat(db: Database, vector_store: VectorStore, tmp_path: Path) -> None:
    pipeline = _pipeline(db, vector_store)
    path = _write_csv(tmp_path)
    first = pipeline.ingest_csv(path)
    second = pipeline.ingest_csv(path)
    assert first.status == "indexed"
    assert second.status == "skipped"
    assert vector_store.count() == first.chunk_count


def test_ingest_csv_force_reindexes(db: Database, vector_store: VectorStore, tmp_path: Path) -> None:
    pipeline = _pipeline(db, vector_store)
    path = _write_csv(tmp_path)
    first = pipeline.ingest_csv(path)
    second = pipeline.ingest_csv(path, force=True)
    assert first.status == "indexed"
    assert second.status == "indexed"
    assert vector_store.count() == first.chunk_count


def test_ingest_csv_non_csv_rejected(db: Database, vector_store: VectorStore, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a csv", encoding="utf-8")
    with pytest.raises(IngestionError, match="Not a CSV file"):
        _pipeline(db, vector_store).ingest_csv(path)


# --------------------------------------------------------------------- #
# SQL ingestion
# --------------------------------------------------------------------- #
def _write_sqlite(tmp_path: Path) -> Path:
    path = tmp_path / "workshop.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE parts (part_number TEXT, name TEXT, unit_price REAL)")
    conn.executemany(
        "INSERT INTO parts VALUES (?, ?, ?)",
        [("BP-101", "Brake pads", 45.5), ("OF-202", "Oil filter", 8.25)],
    )
    conn.commit()
    conn.close()
    return path


def test_ingest_sql_end_to_end(db: Database, vector_store: VectorStore, tmp_path: Path) -> None:
    result = _pipeline(db, vector_store).ingest_sql(str(_write_sqlite(tmp_path)), table="parts")
    assert result.status == "indexed"
    assert result.chunk_count == 2
    assert result.metadata["doc_type"] == "tabular"
    assert result.path.startswith("sqlite://")

    record = DocumentRepository(db).get_by_id(result.document_id)
    assert record is not None
    assert record.status == "indexed"
    assert record.doc_type == "tabular"
    assert record.page_count == 2
    assert record.source_path.startswith("sqlite://")
    assert vector_store.count() == 2


def test_ingest_sql_skips_unchanged_on_repeat(
    db: Database, vector_store: VectorStore, tmp_path: Path
) -> None:
    pipeline = _pipeline(db, vector_store)
    source = str(_write_sqlite(tmp_path))
    first = pipeline.ingest_sql(source, table="parts")
    second = pipeline.ingest_sql(source, table="parts")
    assert first.status == "indexed"
    assert second.status == "skipped"
    assert vector_store.count() == first.chunk_count


def test_ingest_sql_force_reindexes(
    db: Database, vector_store: VectorStore, tmp_path: Path
) -> None:
    pipeline = _pipeline(db, vector_store)
    source = str(_write_sqlite(tmp_path))
    first = pipeline.ingest_sql(source, table="parts")
    second = pipeline.ingest_sql(source, table="parts", force=True)
    assert first.status == "indexed"
    assert second.status == "indexed"
    assert vector_store.count() == first.chunk_count


def test_ingest_sql_custom_query_and_limit(
    db: Database, vector_store: VectorStore, tmp_path: Path
) -> None:
    source = str(_write_sqlite(tmp_path))
    result = _pipeline(db, vector_store).ingest_sql(
        source, query="SELECT name FROM parts", limit=1
    )
    assert result.status == "indexed"
    assert result.chunk_count == 1
    assert result.metadata["doc_type"] == "tabular"
