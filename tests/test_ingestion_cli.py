"""Tests for the ingestion CLI argument parsing and source validation."""

from __future__ import annotations

import pytest

from auto_rag.ingestion.cli import _validate_source_args, build_parser


def _parse(*argv: str):
    return build_parser().parse_args(list(argv))


def test_parser_reads_file_source_flags() -> None:
    args = _parse("--sqlite", "workshop.db", "--table", "parts", "--limit", "5", "--force")
    assert str(args.sqlite) == "workshop.db"
    assert args.table == "parts"
    assert args.limit == 5
    assert args.force is True


def test_parser_reads_sql_url_and_query() -> None:
    args = _parse("--sql-url", "postgresql://u:p@h/db", "--query", "SELECT 1")
    assert args.sql_url == "postgresql://u:p@h/db"
    assert args.query == "SELECT 1"


def test_parser_reads_csv_and_doc_type() -> None:
    args = _parse("--csv", "parts.csv", "--doc-type", "tabular")
    assert str(args.csv) == "parts.csv"
    assert args.doc_type == "tabular"


def test_validate_accepts_single_source() -> None:
    _validate_source_args(_parse("--file", "manual.pdf"))
    _validate_source_args(_parse("--sqlite", "x.db", "--table", "parts"))
    _validate_source_args(_parse("--sql-url", "postgresql://h/db", "--query", "SELECT 1"))
    _validate_source_args(_parse("--directory", "data", "--doc-type", "dtc"))


def test_validate_rejects_sql_combined_with_file() -> None:
    with pytest.raises(SystemExit):
        _validate_source_args(_parse("--sqlite", "x.db", "--file", "y.pdf"))


def test_validate_rejects_sql_combined_with_csv() -> None:
    with pytest.raises(SystemExit):
        _validate_source_args(_parse("--sql-url", "postgresql://h/db", "--csv", "p.csv"))


def test_validate_rejects_no_source() -> None:
    with pytest.raises(SystemExit):
        _validate_source_args(_parse())


def test_validate_rejects_table_without_sql() -> None:
    with pytest.raises(SystemExit):
        _validate_source_args(_parse("--file", "manual.pdf", "--table", "parts"))


def test_validate_rejects_limit_without_sql() -> None:
    with pytest.raises(SystemExit):
        _validate_source_args(_parse("--csv", "parts.csv", "--limit", "5"))
