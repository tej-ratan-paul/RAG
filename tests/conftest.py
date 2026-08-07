"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_rag.config import Settings
from auto_rag.db.connection import Database
from auto_rag.ingestion.embeddings import DeterministicHashEmbeddingProvider
from auto_rag.ingestion.vectorstore import VectorStore


@pytest.fixture
def settings() -> Settings:
    """A Settings instance isolated from any project-level .env file."""
    return Settings(_env_file=None)


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """An isolated sandbox directory used as a fake project root."""
    return tmp_path


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """An initialised, isolated SQLite database."""
    database = Database(path=tmp_path / "test.db")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def deterministic_embeddings() -> DeterministicHashEmbeddingProvider:
    """A fast, deterministic embedding provider for offline tests."""
    return DeterministicHashEmbeddingProvider(dimension=64)


@pytest.fixture
def vector_store(tmp_path: Path, deterministic_embeddings) -> VectorStore:
    """An isolated Chroma collection backed by deterministic embeddings."""
    return VectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_docs",
        provider=deterministic_embeddings,
        distance="cosine",
    )


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Factory producing a small PDF with the given per-page texts."""

    def _make(pages: list[str], name: str = "sample.pdf") -> Path:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        path = tmp_path / name
        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        for text in pages:
            c.drawString(72, height - 72, text)
            c.showPage()
        c.save()
        return path

    return _make
