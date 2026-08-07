"""Tests for the SQLite connection wrapper and migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from auto_rag.db.connection import Database
from auto_rag.db.migrations import MIGRATIONS
from auto_rag.errors import DatabaseError


def test_initialize_creates_all_tables(db: Database) -> None:
    expected = {
        "schema_migrations",
        "vehicles",
        "parts",
        "labor_operations",
        "service_history",
        "maintenance_schedule",
        "dtc_codes",
        "documents",
        "conversations",
        "messages",
    }
    tables = {
        row["name"]
        for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert expected.issubset(tables)


def test_initialize_is_idempotent(db: Database) -> None:
    db.initialize()
    db.initialize()
    count = db.scalar("SELECT COUNT(*) FROM schema_migrations")
    assert count == len(MIGRATIONS)


def test_migrations_recorded_with_descriptions(db: Database) -> None:
    rows = db.query("SELECT version, description FROM schema_migrations ORDER BY version")
    assert [row["version"] for row in rows] == [m.version for m in MIGRATIONS]
    assert all(row["description"] for row in rows)


def test_wal_journal_mode_enabled(tmp_path: Path) -> None:
    path = tmp_path / "wal.db"
    db = Database(path=path, journal_mode="wal")
    db.initialize()
    mode = db.scalar("PRAGMA journal_mode")
    assert mode == "wal"
    db.close()


def test_foreign_keys_enforced(db: Database) -> None:
    db.execute(
        "INSERT INTO vehicles (make, model, year) VALUES ('Toyota', 'Camry', 2018)"
    )
    db.execute(
        """
        INSERT INTO service_history (vehicle_id, service_date, description)
        VALUES (1, '2026-01-01', 'test')
        """
    )
    with pytest.raises(DatabaseError):
        db.execute(
            """
            INSERT INTO service_history (vehicle_id, service_date, description)
            VALUES (999, '2026-01-01', 'orphan')
            """
        )


def test_transaction_rolls_back_on_error(db: Database) -> None:
    db.execute("INSERT INTO vehicles (make, model, year) VALUES ('Honda', 'Civic', 2016)")
    with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
        conn.execute("INSERT INTO vehicles (make, model, year) VALUES ('A', 'B', 2020)")
        conn.execute("INSERT INTO vehicles (make, model, year) VALUES ('C', 'D', 2020)")
        raise sqlite3.IntegrityError("boom")
    count = db.scalar("SELECT COUNT(*) FROM vehicles")
    assert count == 1


def test_execute_returns_lastrowid(db: Database) -> None:
    vehicle_id = db.execute(
        "INSERT INTO vehicles (make, model, year) VALUES ('Toyota', 'Camry', 2021)"
    )
    assert vehicle_id == 1
    assert db.scalar("SELECT COUNT(*) FROM vehicles WHERE id = ?", (vehicle_id,)) == 1


def test_query_one_returns_none_when_empty(db: Database) -> None:
    assert db.query_one("SELECT * FROM vehicles WHERE id = -1") is None


def test_scalar_returns_single_value(db: Database) -> None:
    assert db.scalar("SELECT 1") == 1
    assert db.scalar("SELECT * FROM vehicles LIMIT 1") is None


def test_connect_failure_raises_database_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("i am a file, not a directory")
    bad_path = blocker / "x.db"
    db = Database(path=bad_path)
    with pytest.raises(DatabaseError):
        db._connect()


def test_duplicate_unique_constraint_raises_database_error(db: Database) -> None:
    db.execute("INSERT INTO vehicles (make, model, year, vin) VALUES ('A', 'B', 2020, 'X1')")
    with pytest.raises(DatabaseError):
        db.execute("INSERT INTO vehicles (make, model, year, vin) VALUES ('C', 'D', 2020, 'X1')")
