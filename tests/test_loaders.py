"""Tests for document loaders."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_rag.ingestion.loaders as loaders
from auto_rag.errors import IngestionError
from auto_rag.ingestion.loaders import (
    CSVLoader,
    PDFLoader,
    SQLLoader,
    TextLoader,
    _redact_url,
    loader_for,
    sql_source_label,
)


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
    assert isinstance(loader_for(Path("d.csv")), CSVLoader)


def test_loader_for_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        loader_for(Path("archive.zip"))


# --------------------------------------------------------------------- #
# CSV loader
# --------------------------------------------------------------------- #
def _write_csv(tmp_path: Path, content: str, name: str = "parts.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_csv_loader_rows_become_pages(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "part_number,name,unit_price\n"
        "BP-101,Brake pads,45.50\n"
        "OF-202,Oil filter,8.25\n",
    )
    pages = CSVLoader().load(path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "part_number: BP-101" in pages[0].text
    assert "name: Brake pads" in pages[0].text
    assert pages[0].metadata["columns"] == ["part_number", "name", "unit_price"]
    assert pages[1].text.startswith("part_number: OF-202")


def test_csv_loader_handles_bom_and_quoting(tmp_path: Path) -> None:
    path = tmp_path / "codes.csv"
    path.write_bytes(b"\xef\xbb\xbfcode,description\n" b'"P0300","Random misfire, multiple cylinders"\n')
    pages = CSVLoader().load(path)
    assert len(pages) == 1
    assert "Random misfire, multiple cylinders" in pages[0].text


def test_csv_loader_missing_header_raises(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "")
    with pytest.raises(IngestionError, match="no header"):
        CSVLoader().load(path)


def test_csv_loader_empty_rows_raise(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "a,b\n")
    with pytest.raises(IngestionError, match="no data rows"):
        CSVLoader().load(path)


def test_csv_loader_non_utf8_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff\xfe\x00broken")
    with pytest.raises(IngestionError, match="not valid UTF-8"):
        CSVLoader().load(path)


def test_csv_loader_skips_empty_cells(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "a,b\n1,\n")
    pages = CSVLoader().load(path)
    assert pages[0].text == "a: 1"


# --------------------------------------------------------------------- #
# SQL sources
# --------------------------------------------------------------------- #
def _make_sqlite(tmp_path: Path, name: str = "workshop.db") -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE parts (part_number TEXT, name TEXT, unit_price REAL)"
    )
    conn.executemany(
        "INSERT INTO parts VALUES (?, ?, ?)",
        [("BP-101", "Brake pads", 45.5), ("OF-202", "Oil filter", 8.25)],
    )
    conn.commit()
    conn.close()
    return path


def test_sql_loader_table_select(tmp_path: Path) -> None:
    path = _make_sqlite(tmp_path)
    pages = SQLLoader(table="parts").load_source(str(path))
    assert len(pages) == 2
    assert "part_number: BP-101" in pages[0].text
    assert pages[0].metadata["columns"] == ["part_number", "name", "unit_price"]


def test_sql_loader_custom_query(tmp_path: Path) -> None:
    path = _make_sqlite(tmp_path)
    pages = SQLLoader(query="SELECT name, unit_price FROM parts").load_source(str(path))
    assert len(pages) == 2
    assert "name: Oil filter" in pages[1].text


def test_sql_loader_limit(tmp_path: Path) -> None:
    path = _make_sqlite(tmp_path)
    pages = SQLLoader(table="parts", limit=1).load_source(str(path))
    assert len(pages) == 1


def test_sql_loader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        SQLLoader(table="parts").load_source(str(tmp_path / "nope.db"))


def test_sql_loader_requires_table_or_query() -> None:
    with pytest.raises(IngestionError, match="--table or --query"):
        SQLLoader()


def test_sql_loader_bad_table_raises(tmp_path: Path) -> None:
    path = _make_sqlite(tmp_path)
    with pytest.raises(IngestionError, match="no such table"):
        SQLLoader(table="missing").load_source(str(path))


def test_sql_loader_remote_missing_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name, *args, **kwargs):
        raise ImportError(name)

    monkeypatch.setattr(loaders, "__import__", fake_import, raising=False)
    with pytest.raises(IngestionError, match="psycopg2"):
        SQLLoader(table="parts").load_source("postgresql://user:pass@host/db")


def test_sql_loader_remote_uses_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cursor:
        def __init__(self, rows):
            self._rows = rows
            self.description = [("code",), ("severity",)]

        def execute(self, sql):
            pass

        def fetchall(self):
            return self._rows

        def close(self):
            pass

    class _Connection:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self):
            return _Cursor(self._rows)

        def close(self):
            pass

    calls: list[str] = []

    def fake_connect(url):
        calls.append(url)
        return _Connection([("P0300", "high")])

    def fake_import(name, *args, **kwargs):
        return SimpleNamespace(connect=fake_connect)

    monkeypatch.setattr(loaders, "__import__", fake_import, raising=False)
    pages = SQLLoader(table="dtc_codes").load_source("mysql://user:pass@host/db")
    assert len(pages) == 1
    assert "code: P0300" in pages[0].text
    assert "severity: high" in pages[0].text


def test_sql_loader_unsupported_scheme() -> None:
    with pytest.raises(IngestionError, match="Unsupported SQL URL scheme"):
        SQLLoader(table="x").load_source("oracle://host/db")


# --------------------------------------------------------------------- #
# Source labels
# --------------------------------------------------------------------- #
def test_sql_source_label_for_file() -> None:
    label = sql_source_label(r"C:\data\parts.db", table="parts")
    assert label.startswith("sqlite://C:\\data\\parts.db")
    assert "?table=parts" in label


def test_sql_source_label_redacts_credentials() -> None:
    label = sql_source_label("postgresql://user:secret@host:5432/db", table="vehicles")
    assert "secret" not in label
    assert label.startswith("postgresql://")
    assert "?table=vehicles" in label


def test_redact_url() -> None:
    assert _redact_url("postgres://u:p@h/db") == "postgres://h/db"
    assert _redact_url("plain path") == "plain path"
