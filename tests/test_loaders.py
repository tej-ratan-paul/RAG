"""Tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_rag.errors import IngestionError
from auto_rag.ingestion.loaders import PDFLoader, TextLoader, loader_for


def test_pdf_loader_extracts_pages(make_pdf) -> None:
    path = make_pdf(
        [
            "P0300 random misfire service steps.",
            "Check the spark plugs for wear.",
        ]
    )
    pages = PDFLoader().load(path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "P0300" in pages[0].text
    assert pages[1].page_number == 2


def test_text_loader_single_page(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Some plain notes.", encoding="utf-8")
    pages = TextLoader().load(path)
    assert len(pages) == 1
    assert pages[0].text == "Some plain notes."


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        PDFLoader().load(tmp_path / "missing.pdf")


def test_loader_for_dispatches_on_suffix(tmp_path: Path) -> None:
    assert isinstance(loader_for(Path("a.pdf")), PDFLoader)
    assert isinstance(loader_for(Path("b.txt")), TextLoader)
    assert isinstance(loader_for(Path("c.md")), TextLoader)


def test_loader_for_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        loader_for(Path("archive.zip"))
