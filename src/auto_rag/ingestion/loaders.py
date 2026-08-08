"""Document loaders.

Extracts raw text from source files into a list of :class:`PageContent`
objects, preserving page boundaries and per-page raw metadata. Supports
PDF, plain text, CSV, and SQL (SQLite files or remote SQL URLs).
"""

from __future__ import annotations

import csv
import io
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from auto_rag.errors import IngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageContent:
    """Raw text and metadata extracted from a single page."""

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Protocol implemented by all loaders."""

    def load(self, path: Path) -> list[PageContent]:
        """Return one :class:`PageContent` per page in the document."""
        ...


class PDFLoader:
    """Loads text from PDF files using pypdf."""

    def load(self, path: Path) -> list[PageContent]:
        """Extract per-page text from ``path``."""
        if not path.is_file():
            raise IngestionError(f"PDF not found: {path}")
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF {path}: {exc}") from exc

        if reader.is_encrypted:
            raise IngestionError(
                f"Encrypted PDF not supported: {path} (please decrypt before ingestion)"
            )

        pages: list[PageContent] = []
        for index, page in enumerate(reader.pages, start=1):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - pypdf internals vary
                logger.warning("Page %d of %s could not be extracted: %s", index, path, exc)
            pages.append(
                PageContent(page_number=index, text=text, metadata={"page_number": index})
            )
        logger.debug("Loaded %d pages from %s", len(pages), path)
        return pages


class TextLoader:
    """Loads plain-text files (``.txt``, ``.md``) as a single page."""

    def load(self, path: Path) -> list[PageContent]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestionError(f"Failed to read text file {path}: {exc}") from exc
        return [PageContent(page_number=1, text=text, metadata={"page_number": 1})]


class CSVLoader:
    """Loads CSV files as one row per page.

    Each row becomes a single :class:`PageContent` whose text is formatted as
    ``column: value`` pairs, so every row stays intact as one chunk. The header
    row defines the column names, stored in the page metadata.
    """

    def load(self, path: Path) -> list[PageContent]:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise IngestionError(f"Failed to read CSV file {path}: {exc}") from exc

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                f"CSV {path} is not valid UTF-8 (only UTF-8 CSV is supported)"
            ) from exc

        try:
            reader = csv.DictReader(io.StringIO(text))
            fieldnames = reader.fieldnames
            rows = list(reader)
        except csv.Error as exc:
            raise IngestionError(f"Failed to parse CSV {path}: {exc}") from exc

        if not fieldnames:
            raise IngestionError(f"CSV {path} has no header row")

        pages: list[PageContent] = []
        for index, row in enumerate(rows, start=1):
            parts = [
                f"{column}: {row.get(column)}"
                for column in fieldnames
                if row.get(column) not in (None, "")
            ]
            if not parts:
                continue
            pages.append(
                PageContent(
                    page_number=index,
                    text=", ".join(parts),
                    metadata={
                        "page_number": index,
                        "row": index,
                        "columns": fieldnames,
                    },
                )
            )
        if not pages:
            raise IngestionError(f"CSV {path} contains no data rows")
        logger.debug("Loaded %d rows from %s", len(pages), path)
        return pages


# --------------------------------------------------------------------- #
# SQL sources
# --------------------------------------------------------------------- #
_SQL_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*)://", re.IGNORECASE)

_REMOTE_DRIVERS: dict[str, str] = {
    "postgres": "psycopg2",
    "postgresql": "psycopg2",
    "mysql": "pymysql",
    "mariadb": "pymysql",
}

_DEFAULT_SQL_QUERY = "SELECT * FROM {table}"


def _redact_url(url: str) -> str:
    """Return a URL with any embedded credentials removed."""
    match = _SQL_SCHEME.match(url)
    if match is None:
        return url
    remainder = url[match.end() :]
    head, separator, _tail = remainder.partition("@")
    if not separator:
        return url
    return f"{match.group(1)}://{_tail}"


def _url_is_remote(url: str) -> bool:
    return bool(_SQL_SCHEME.match(url))


def sql_source_label(
    source: str,
    *,
    table: str | None = None,
    query: str | None = None,
) -> str:
    """Build a stable, human-readable identity label for a SQL source.

    For file sources the label is ``sqlite://<path>``; for remote sources the
    credentials are redacted. Table/query names are appended as a query string.
    """
    label = source
    if _url_is_remote(source):
        label = _redact_url(source)
    elif _SQL_SCHEME.match(source) is None:
        label = f"sqlite://{source}"
    parts: list[str] = []
    if table:
        parts.append(f"table={table}")
    if query:
        parts.append("query=<custom>")
    suffix = "?" + "&".join(parts) if parts else ""
    return f"{label}{suffix}"


def _quote_ident(identifier: str) -> str:
    """Quote a SQL identifier for safe interpolation into a query."""
    return '"' + identifier.replace('"', '""') + '"'


class SQLLoader:
    """Loads rows from a SQLite database file or a remote SQL URL.

    Args:
        table: Table to read (``SELECT * FROM <table>``).
        query: Raw SQL query. If both are given, ``query`` wins.
        limit: Optional row cap applied to the fetched result.
    """

    def __init__(
        self,
        *,
        table: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> None:
        if not table and not query:
            raise IngestionError("A SQL source needs either --table or --query")
        self.table = table
        self.query = query
        self.limit = limit

    def load_source(self, source: str) -> list[PageContent]:
        """Fetch rows from ``source`` (a SQLite file path or remote URL)."""
        if _url_is_remote(source):
            rows = self._load_remote(source)
        else:
            rows = self._load_sqlite(Path(source))
        return self._rows_to_pages(rows)

    # ------------------------------------------------------------------ #
    def _load_sqlite(self, path: Path) -> tuple[str, list[Any]]:
        if not path.is_file():
            raise IngestionError(f"SQLite database not found: {path}")
        try:
            uri = path.resolve().as_uri()
            connection = sqlite3.connect(f"{uri}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                statement = self._build_query(str(path))
                cursor = connection.execute(statement)
                fetched = cursor.fetchall()
                if self.limit is not None:
                    fetched = fetched[: self.limit]
                columns = [name for name, *_ in cursor.description or ()]
                return columns, [list(row) for row in fetched]
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise IngestionError(f"SQLite query failed for {path}: {exc}") from exc

    def _load_remote(self, url: str) -> tuple[str, list[Any]]:
        scheme = _SQL_SCHEME.match(url).group(1).lower()
        driver = _REMOTE_DRIVERS.get(scheme)
        if driver is None:
            raise IngestionError(
                f"Unsupported SQL URL scheme {scheme!r}; "
                f"supported: {', '.join(sorted(_REMOTE_DRIVERS))}"
            )
        try:
            module = __import__(driver)
        except ImportError:
            raise IngestionError(
                f"Remote SQL ingestion for {scheme!r} requires the optional "
                f"driver {driver!r} (e.g. `pip install {driver}`). "
                "SQLite files need no extra dependency."
            ) from None

        try:
            connection = module.connect(url)
            try:
                cursor = connection.cursor()
                cursor.execute(self._build_query(url))
                columns = [desc[0] for desc in cursor.description or ()]
                fetched = cursor.fetchall()
                if self.limit is not None:
                    fetched = fetched[: self.limit]
                return tuple(columns), [list(row) for row in fetched]
            finally:
                cursor.close()
                connection.close()
        except Exception as exc:
            raise IngestionError(f"SQL query failed for {url}: {exc}") from exc

    # ------------------------------------------------------------------ #
    def _build_query(self, source: str) -> str:
        if self.query:
            return self.query
        statement = _DEFAULT_SQL_QUERY.format(table=_quote_ident(self.table))
        if self.limit is not None:
            statement = f"{statement} LIMIT {self.limit}"
        return statement

    def _rows_to_pages(self, data: tuple[str, list[Any]]) -> list[PageContent]:
        columns, rows = data
        pages: list[PageContent] = []
        for index, row in enumerate(rows, start=1):
            parts = [
                f"{column}: {value}"
                for column, value in zip(columns, row, strict=False)
                if value not in (None, "")
            ]
            pages.append(
                PageContent(
                    page_number=index,
                    text=", ".join(parts),
                    metadata={
                        "page_number": index,
                        "row": index,
                        "columns": list(columns),
                    },
                )
            )
        return pages


SUPPORTED_SUFFIXES: dict[str, DocumentLoader] = {
    ".pdf": PDFLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
    ".csv": CSVLoader(),
}


def loader_for(path: Path) -> DocumentLoader:
    """Return the appropriate loader for ``path``'s suffix."""
    loader = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if loader is None:
        raise IngestionError(
            f"Unsupported file type {path.suffix!r} for {path.name} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    return loader
