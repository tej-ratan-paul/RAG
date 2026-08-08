"""Database schema migrations.

Migrations are ordered, immutable, and applied atomically (each migration
runs inside a single transaction recorded in ``schema_migrations``). Never
edit an applied migration; append a new one instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    """A single ordered schema migration."""

    version: str
    description: str
    statements: tuple[str, ...]


_DOCUMENTS_DDL: str = """
CREATE TABLE documents (
    id           INTEGER PRIMARY KEY,
    source_path  TEXT NOT NULL,
    file_hash    TEXT NOT NULL UNIQUE,
    doc_type     TEXT NOT NULL CHECK (
        doc_type IN (
            'service_manual','repair_manual','dtc','tsb','wiring_diagram','tabular'
        )
    ),
    title        TEXT,
    make         TEXT,
    model        TEXT,
    year         INTEGER,
    engine       TEXT,
    vin          TEXT,
    page_count   INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending','indexed','failed')
    ),
    error        TEXT,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DOCUMENTS_COLUMNS: str = (
    "id, source_path, file_hash, doc_type, title, make, model, year, engine, "
    "vin, page_count, chunk_count, status, error, ingested_at, updated_at"
)

_INITIAL_SCHEMA: tuple[str, ...] = (
    # ------------------------------------------------------------------ #
    # Schema bookkeeping
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version     TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # ------------------------------------------------------------------ #
    # Vehicles
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS vehicles (
        id         INTEGER PRIMARY KEY,
        make       TEXT NOT NULL,
        model      TEXT NOT NULL,
        year       INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
        engine     TEXT,
        vin        TEXT UNIQUE,
        mileage    INTEGER CHECK (mileage IS NULL OR mileage >= 0),
        notes      TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_vehicles_make_model
        ON vehicles(make, model, year)
    """,
    # ------------------------------------------------------------------ #
    # Parts inventory
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS parts (
        id                INTEGER PRIMARY KEY,
        part_number       TEXT NOT NULL UNIQUE,
        name              TEXT NOT NULL,
        description       TEXT,
        category          TEXT,
        manufacturer      TEXT,
        compatible_make   TEXT,
        compatible_model  TEXT,
        compatible_years  TEXT,
        quantity_on_hand  INTEGER NOT NULL DEFAULT 0 CHECK (quantity_on_hand >= 0),
        reorder_level     INTEGER NOT NULL DEFAULT 0,
        unit_price        REAL NOT NULL DEFAULT 0.0 CHECK (unit_price >= 0),
        location          TEXT,
        updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_parts_name ON parts(name)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_parts_category ON parts(category)
    """,
    # ------------------------------------------------------------------ #
    # Labor operations / rates
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS labor_operations (
        id             INTEGER PRIMARY KEY,
        operation_code TEXT NOT NULL UNIQUE,
        description    TEXT NOT NULL,
        skill_level    TEXT NOT NULL DEFAULT 'Technician',
        standard_hours REAL NOT NULL CHECK (standard_hours > 0),
        rate_per_hour  REAL NOT NULL CHECK (rate_per_hour >= 0),
        notes          TEXT,
        updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # ------------------------------------------------------------------ #
    # Service history
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS service_history (
        id                   INTEGER PRIMARY KEY,
        vehicle_id           INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
        service_date         TEXT NOT NULL,
        mileage              INTEGER,
        description          TEXT NOT NULL,
        dtc_codes            TEXT,
        parts_used           TEXT,
        labor_operation_code TEXT,
        total_cost           REAL,
        technician           TEXT,
        notes                TEXT,
        created_at           TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_service_history_vehicle
        ON service_history(vehicle_id, service_date DESC)
    """,
    # ------------------------------------------------------------------ #
    # Maintenance schedule
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS maintenance_schedule (
        id              INTEGER PRIMARY KEY,
        task            TEXT NOT NULL,
        description     TEXT,
        make            TEXT,
        model           TEXT,
        year            INTEGER,
        interval_miles  INTEGER CHECK (interval_miles IS NULL OR interval_miles > 0),
        interval_months INTEGER CHECK (interval_months IS NULL OR interval_months > 0),
        parts_required  TEXT,
        priority        TEXT NOT NULL DEFAULT 'standard',
        updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_maintenance_make_model
        ON maintenance_schedule(make, model)
    """,
    # ------------------------------------------------------------------ #
    # Diagnostic trouble codes
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS dtc_codes (
        id               INTEGER PRIMARY KEY,
        code             TEXT NOT NULL UNIQUE,
        description      TEXT NOT NULL,
        system           TEXT,
        manufacturer     TEXT,
        possible_causes  TEXT,
        diagnostic_notes TEXT,
        severity         TEXT NOT NULL DEFAULT 'unknown',
        updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dtc_codes_description ON dtc_codes(description)
    """,
    # ------------------------------------------------------------------ #
    # Source documents (ingestion tracking)
    # ------------------------------------------------------------------ #
    _DOCUMENTS_DDL,
    # ------------------------------------------------------------------ #
    # Conversations (persisted chat + vehicle context for memory)
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id         INTEGER PRIMARY KEY,
        title      TEXT NOT NULL DEFAULT 'New conversation',
        vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
        content         TEXT NOT NULL,
        tool_name       TEXT,
        citations       TEXT,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id, id)
    """,
)

MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="0001_initial",
        description="Initial schema: vehicles, parts, labor, service history, "
        "maintenance, DTCs, documents, conversations.",
        statements=_INITIAL_SCHEMA,
    ),
    Migration(
        version="0002_tabular_doc_type",
        description="Allow doc_type 'tabular' for CSV/SQL ingestion.",
        statements=(
            "ALTER TABLE documents RENAME TO documents_0002_old",
            _DOCUMENTS_DDL,
            (
                "INSERT INTO documents (" + _DOCUMENTS_COLUMNS + ") "
                "SELECT " + _DOCUMENTS_COLUMNS + " FROM documents_0002_old"
            ),
            "DROP TABLE documents_0002_old",
        ),
    ),
)
