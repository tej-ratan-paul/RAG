"""SQLite connection management and schema initialisation.

:class:`Database` provides a thin, safe wrapper around the stdlib
``sqlite3`` module: one connection per operation (thread-safe under WAL),
foreign keys always on, busy-timeout configured, and atomic migration
application.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from auto_rag.config import Settings
from auto_rag.db.migrations import MIGRATIONS, Migration
from auto_rag.errors import DatabaseError
from auto_rag.utils.paths import ensure_directory
from auto_rag.utils.time import utcnow_iso

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 10.0
_DEFAULT_JOURNAL_MODE: str = "wal"


def _is_insert(sql: str) -> bool:
    """Heuristic: whether a statement begins with INSERT (or WITH...INSERT)."""
    stripped = sql.lstrip().upper()
    return stripped.startswith("INSERT") or (
        stripped.startswith("WITH") and " INSERT " in f" {stripped} "
    )


class Database:
    """SQLite database wrapper with transactional migrations.

    Usage::

        db = Database.from_settings(settings)
        db.initialize()
        rows = db.query("SELECT * FROM parts WHERE name LIKE ?", ("%filter%",))
    """

    def __init__(
        self,
        path: Path,
        timeout: float = _DEFAULT_TIMEOUT,
        journal_mode: str = _DEFAULT_JOURNAL_MODE,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.journal_mode = journal_mode
        self._lock = threading.RLock()
        self._initialized = False

    @classmethod
    def from_settings(cls, settings: Settings) -> Database:
        """Construct a :class:`Database` from application settings."""
        return cls(
            path=settings.sqlite_path,
            timeout=settings.database.timeout,
            journal_mode=settings.database.journal_mode,
        )

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        try:
            ensure_directory(self.path.parent)
            conn = sqlite3.connect(
                self.path,
                timeout=self.timeout,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
            return conn
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Could not connect to {self.path}: {exc}") from exc

    def close(self) -> None:
        """Release the thread lock; subsequent calls reopen connections."""
        with self._lock:
            self._initialized = False

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #
    def execute(
        self,
        sql: str,
        params: Sequence[Any] | dict[str, Any] = (),
    ) -> int:
        """Execute a write statement.

        Returns the inserted row id for ``INSERT`` statements, otherwise the
        number of affected rows.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, params)
                if _is_insert(sql) and cursor.lastrowid is not None:
                    return cursor.lastrowid
                return cursor.rowcount
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Execute failed: {exc}") from exc

    def query(
        self,
        sql: str,
        params: Sequence[Any] | dict[str, Any] = (),
    ) -> list[dict[str, Any]]:
        """Execute a read statement and return a list of row dicts."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    def query_one(
        self,
        sql: str,
        params: Sequence[Any] | dict[str, Any] = (),
    ) -> dict[str, Any] | None:
        """Execute a read statement and return a single row dict or None."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
        """Execute a read statement and return the first column value."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                return row[0] if row is not None else None
        except sqlite3.Error as exc:
            raise DatabaseError(f"Scalar query failed: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager providing an explicit transactional connection.

        Commits on success, rolls back on error.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Migrations
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """Apply any pending migrations (idempotent, thread-safe)."""
        with self._lock:
            if self._initialized:
                return
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version     TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                applied = {
                    row["version"]
                    for row in conn.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                for migration in MIGRATIONS:
                    self._apply_migration(conn, migration, applied)
                conn.close()
            except sqlite3.Error as exc:
                conn.close()
                raise DatabaseError(f"Migration failed: {exc}") from exc
            self._initialized = True
            logger.info(
                "Database initialised: path=%s migrations=%d",
                self.path,
                len(MIGRATIONS),
            )

    @staticmethod
    def _apply_migration(
        conn: sqlite3.Connection,
        migration: Migration,
        applied: set[str],
    ) -> None:
        if migration.version in applied:
            return
        logger.debug("Applying migration %s (%s)", migration.version, migration.description)
        try:
            conn.execute("BEGIN")
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.description, utcnow_iso()),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
