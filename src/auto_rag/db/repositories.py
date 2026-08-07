"""Repository layer: typed data access over :class:`Database`.

Each repository owns the queries for a single domain concept and converts
rows to/from the Pydantic models in :mod:`auto_rag.db.models`. Repositories
take a :class:`Database` via constructor injection.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from auto_rag.db.connection import Database
from auto_rag.db.models import (
    Conversation,
    DocumentRecord,
    DTCCode,
    LaborOperation,
    MaintenanceTask,
    Message,
    Part,
    ServiceRecord,
    Vehicle,
)
from auto_rag.utils.time import utcnow_iso

logger = logging.getLogger(__name__)


class _BaseRepository:
    """Shared plumbing for repositories."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _like(value: str) -> str:
        return f"%{value.strip()}%"


class VehicleRepository(_BaseRepository):
    """CRUD for vehicles."""

    _COLUMNS = "id, make, model, year, engine, vin, mileage, notes, created_at, updated_at"

    def get_by_id(self, vehicle_id: int) -> Vehicle | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM vehicles WHERE id = ?", (vehicle_id,)
        )
        return Vehicle(**row) if row else None

    def get_by_vin(self, vin: str) -> Vehicle | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM vehicles WHERE vin = ?", (vin.strip().upper(),)
        )
        return Vehicle(**row) if row else None

    def upsert(self, vehicle: Vehicle) -> int:
        """Insert or update by VIN; returns the vehicle id."""
        vin = vehicle.vin.strip().upper() if vehicle.vin else None
        existing = self.get_by_vin(vin) if vin else None
        if existing is not None:
            self.db.execute(
                """
                UPDATE vehicles
                   SET make = ?, model = ?, year = ?, engine = ?, vin = ?,
                       mileage = ?, notes = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    vehicle.make,
                    vehicle.model,
                    vehicle.year,
                    vehicle.engine,
                    vin,
                    vehicle.mileage,
                    vehicle.notes,
                    utcnow_iso(),
                    existing.id,
                ),
            )
            return existing.id
        return self.db.execute(
            """
            INSERT INTO vehicles (make, model, year, engine, vin, mileage, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vehicle.make,
                vehicle.model,
                vehicle.year,
                vehicle.engine,
                vin,
                vehicle.mileage,
                vehicle.notes,
            ),
        )

    def search(
        self,
        make: str | None = None,
        model: str | None = None,
        year: int | None = None,
        limit: int = 20,
    ) -> list[Vehicle]:
        conditions: list[str] = []
        params: list[Any] = []
        if make:
            conditions.append("make = ?")
            params.append(make)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if year:
            conditions.append("year = ?")
            params.append(year)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM vehicles {where} "
            "ORDER BY year DESC, make, model LIMIT ?",
            (*params, limit),
        )
        return [Vehicle(**row) for row in rows]

    def list_all(self, limit: int = 100) -> list[Vehicle]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM vehicles ORDER BY make, model, year DESC LIMIT ?",
            (limit,),
        )
        return [Vehicle(**row) for row in rows]

    def delete(self, vehicle_id: int) -> bool:
        return self.db.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,)) > 0


class PartRepository(_BaseRepository):
    """CRUD and search for the parts inventory."""

    _COLUMNS = (
        "id, part_number, name, description, category, manufacturer, "
        "compatible_make, compatible_model, compatible_years, "
        "quantity_on_hand, reorder_level, unit_price, location, updated_at"
    )

    def get_by_part_number(self, part_number: str) -> Part | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM parts WHERE part_number = ?",
            (part_number.strip().upper(),),
        )
        return Part(**row) if row else None

    def upsert(self, part: Part) -> int:
        existing = self.get_by_part_number(part.part_number)
        if existing is not None:
            self.db.execute(
                """
                UPDATE parts
                   SET name = ?, description = ?, category = ?, manufacturer = ?,
                       compatible_make = ?, compatible_model = ?, compatible_years = ?,
                       quantity_on_hand = ?, reorder_level = ?, unit_price = ?,
                       location = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    part.name,
                    part.description,
                    part.category,
                    part.manufacturer,
                    part.compatible_make,
                    part.compatible_model,
                    part.compatible_years,
                    part.quantity_on_hand,
                    part.reorder_level,
                    part.unit_price,
                    part.location,
                    utcnow_iso(),
                    existing.id,
                ),
            )
            return existing.id
        return self.db.execute(
            """
            INSERT INTO parts (part_number, name, description, category, manufacturer,
                               compatible_make, compatible_model, compatible_years,
                               quantity_on_hand, reorder_level, unit_price, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part.part_number.strip().upper(),
                part.name,
                part.description,
                part.category,
                part.manufacturer,
                part.compatible_make,
                part.compatible_model,
                part.compatible_years,
                part.quantity_on_hand,
                part.reorder_level,
                part.unit_price,
                part.location,
            ),
        )

    def update_stock(self, part_number: str, delta: int) -> Part | None:
        """Adjust stock by ``delta`` and return the updated part."""
        self.db.execute(
            """
            UPDATE parts
               SET quantity_on_hand = MAX(0, quantity_on_hand + ?),
                   updated_at = ?
             WHERE part_number = ?
            """,
            (delta, utcnow_iso(), part_number.strip().upper()),
        )
        return self.get_by_part_number(part_number)

    def search(
        self,
        query: str,
        make: str | None = None,
        model: str | None = None,
        limit: int = 20,
    ) -> list[Part]:
        conditions = ["(part_number LIKE ? OR name LIKE ? OR description LIKE ?)"]
        params: list[Any] = [self._like(query), self._like(query), self._like(query)]
        if make:
            conditions.append("(compatible_make IS NULL OR compatible_make = ?)")
            params.append(make)
        if model:
            conditions.append("(compatible_model IS NULL OR compatible_model = ?)")
            params.append(model)
        where = " AND ".join(conditions)
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM parts WHERE {where} "
            "ORDER BY name LIMIT ?",
            (*params, limit),
        )
        return [Part(**row) for row in rows]


class LaborRepository(_BaseRepository):
    """Access to labor operations and rates."""

    _COLUMNS = (
        "id, operation_code, description, skill_level, standard_hours, "
        "rate_per_hour, notes, updated_at"
    )

    def get_by_operation_code(self, operation_code: str) -> LaborOperation | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM labor_operations WHERE operation_code = ?",
            (operation_code.strip().upper(),),
        )
        return LaborOperation(**row) if row else None

    def upsert(self, operation: LaborOperation) -> int:
        existing = self.get_by_operation_code(operation.operation_code)
        if existing is not None:
            self.db.execute(
                """
                UPDATE labor_operations
                   SET description = ?, skill_level = ?, standard_hours = ?,
                       rate_per_hour = ?, notes = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    operation.description,
                    operation.skill_level,
                    operation.standard_hours,
                    operation.rate_per_hour,
                    operation.notes,
                    utcnow_iso(),
                    existing.id,
                ),
            )
            return existing.id
        return self.db.execute(
            """
            INSERT INTO labor_operations
                (operation_code, description, skill_level, standard_hours, rate_per_hour, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                operation.operation_code.strip().upper(),
                operation.description,
                operation.skill_level,
                operation.standard_hours,
                operation.rate_per_hour,
                operation.notes,
            ),
        )

    def estimate_cost(self, operation_code: str) -> float | None:
        """Return labour cost for an operation code (hours x rate) or None."""
        operation = self.get_by_operation_code(operation_code)
        return operation.estimated_cost if operation else None

    def search(self, query: str, limit: int = 20) -> list[LaborOperation]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM labor_operations "
            "WHERE operation_code LIKE ? OR description LIKE ? "
            "ORDER BY operation_code LIMIT ?",
            (self._like(query), self._like(query), limit),
        )
        return [LaborOperation(**row) for row in rows]


class ServiceHistoryRepository(_BaseRepository):
    """CRUD for vehicle service history."""

    _COLUMNS = (
        "id, vehicle_id, service_date, mileage, description, dtc_codes, "
        "parts_used, labor_operation_code, total_cost, technician, notes, created_at"
    )

    def add(self, record: ServiceRecord) -> int:
        return self.db.execute(
            """
            INSERT INTO service_history
                (vehicle_id, service_date, mileage, description, dtc_codes,
                 parts_used, labor_operation_code, total_cost, technician, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.vehicle_id,
                record.service_date,
                record.mileage,
                record.description,
                record.dtc_codes,
                record.parts_used,
                record.labor_operation_code,
                record.total_cost,
                record.technician,
                record.notes,
            ),
        )

    def list_by_vehicle(self, vehicle_id: int, limit: int = 50) -> list[ServiceRecord]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM service_history WHERE vehicle_id = ? "
            "ORDER BY service_date DESC LIMIT ?",
            (vehicle_id, limit),
        )
        return [ServiceRecord(**row) for row in rows]

    def recent(self, limit: int = 50) -> list[ServiceRecord]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM service_history "
            "ORDER BY service_date DESC LIMIT ?",
            (limit,),
        )
        return [ServiceRecord(**row) for row in rows]


class MaintenanceRepository(_BaseRepository):
    """Access to the maintenance schedule."""

    _COLUMNS = (
        "id, task, description, make, model, year, interval_miles, "
        "interval_months, parts_required, priority, updated_at"
    )

    def upsert(self, task: MaintenanceTask) -> int:
        existing = self.db.query_one(
            """
            SELECT id FROM maintenance_schedule
            WHERE task = ? AND (make IS ? OR make = ?) AND (model IS ? OR model = ?)
            """,
            (
                task.task,
                task.make,
                task.make,
                task.model,
                task.model,
            ),
        )
        if existing:
            self.db.execute(
                """
                UPDATE maintenance_schedule
                   SET description = ?, make = ?, model = ?, year = ?,
                       interval_miles = ?, interval_months = ?, parts_required = ?,
                       priority = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    task.description,
                    task.make,
                    task.model,
                    task.year,
                    task.interval_miles,
                    task.interval_months,
                    task.parts_required,
                    task.priority,
                    utcnow_iso(),
                    existing["id"],
                ),
            )
            return existing["id"]
        return self.db.execute(
            """
            INSERT INTO maintenance_schedule
                (task, description, make, model, year, interval_miles,
                 interval_months, parts_required, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task,
                task.description,
                task.make,
                task.model,
                task.year,
                task.interval_miles,
                task.interval_months,
                task.parts_required,
                task.priority,
            ),
        )

    def list_for_vehicle(self, vehicle: Vehicle) -> list[MaintenanceTask]:
        """Tasks applicable to a vehicle (NULL make/model = universal)."""
        rows = self.db.query(
            f"""
            SELECT {self._COLUMNS} FROM maintenance_schedule
            WHERE (make IS NULL OR make = ?)
              AND (model IS NULL OR model = ?)
              AND (year IS NULL OR year = ?)
            ORDER BY COALESCE(interval_miles, 999999), COALESCE(interval_months, 999999)
            """,
            (vehicle.make, vehicle.model, vehicle.year),
        )
        return [MaintenanceTask(**row) for row in rows]


class DTCCodeRepository(_BaseRepository):
    """CRUD and search for diagnostic trouble codes."""

    _COLUMNS = (
        "id, code, description, system, manufacturer, possible_causes, "
        "diagnostic_notes, severity, updated_at"
    )

    def get_by_code(self, code: str) -> DTCCode | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM dtc_codes WHERE code = ?",
            (code.strip().upper(),),
        )
        return DTCCode(**row) if row else None

    def upsert(self, dtc: DTCCode) -> int:
        existing = self.get_by_code(dtc.code)
        if existing is not None:
            self.db.execute(
                """
                UPDATE dtc_codes
                   SET description = ?, system = ?, manufacturer = ?,
                       possible_causes = ?, diagnostic_notes = ?, severity = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    dtc.description,
                    dtc.system,
                    dtc.manufacturer,
                    dtc.possible_causes,
                    dtc.diagnostic_notes,
                    dtc.severity,
                    utcnow_iso(),
                    existing.id,
                ),
            )
            return existing.id
        return self.db.execute(
            """
            INSERT INTO dtc_codes
                (code, description, system, manufacturer, possible_causes,
                 diagnostic_notes, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dtc.code.strip().upper(),
                dtc.description,
                dtc.system,
                dtc.manufacturer,
                dtc.possible_causes,
                dtc.diagnostic_notes,
                dtc.severity,
            ),
        )

    def search(self, query: str, limit: int = 20) -> list[DTCCode]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM dtc_codes "
            "WHERE code LIKE ? OR description LIKE ? OR possible_causes LIKE ? "
            "ORDER BY code LIMIT ?",
            (self._like(query), self._like(query), self._like(query), limit),
        )
        return [DTCCode(**row) for row in rows]

    def list_by_system(self, system: str, limit: int = 50) -> list[DTCCode]:
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM dtc_codes WHERE system = ? "
            "ORDER BY code LIMIT ?",
            (system, limit),
        )
        return [DTCCode(**row) for row in rows]


class DocumentRepository(_BaseRepository):
    """Track source documents through the ingestion pipeline."""

    _COLUMNS = (
        "id, source_path, file_hash, doc_type, title, make, model, year, "
        "engine, vin, page_count, chunk_count, status, error, ingested_at, updated_at"
    )

    def get_by_id(self, document_id: int) -> DocumentRecord | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM documents WHERE id = ?", (document_id,)
        )
        return DocumentRecord(**row) if row else None

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM documents WHERE file_hash = ?", (file_hash,)
        )
        return DocumentRecord(**row) if row else None

    def add(self, record: DocumentRecord) -> int:
        return self.db.execute(
            """
            INSERT INTO documents (source_path, file_hash, doc_type, title, make,
                                   model, year, engine, vin, page_count, chunk_count,
                                   status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.source_path,
                record.file_hash,
                record.doc_type,
                record.title,
                record.make,
                record.model,
                record.year,
                record.engine,
                record.vin,
                record.page_count,
                record.chunk_count,
                record.status,
                record.error,
            ),
        )

    def update_status(
        self,
        document_id: int,
        status: str,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        fields = "status = ?, updated_at = ?"
        params: list[Any] = [status, utcnow_iso()]
        if chunk_count is not None:
            fields += ", chunk_count = ?"
            params.append(chunk_count)
        if error is not None:
            fields += ", error = ?"
            params.append(error)
        params.append(document_id)
        self.db.execute(
            f"UPDATE documents SET {fields} WHERE id = ?",
            tuple(params),
        )

    def update_metadata(
        self,
        document_id: int,
        *,
        doc_type: str | None = None,
        title: str | None = None,
        make: str | None = None,
        model: str | None = None,
        year: int | None = None,
        engine: str | None = None,
        vin: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """Persist extracted metadata onto a document record."""
        fields = ["updated_at = ?"]
        params: list[Any] = [utcnow_iso()]
        for column, value in (
            ("doc_type", doc_type),
            ("title", title),
            ("make", make),
            ("model", model),
            ("year", year),
            ("engine", engine),
            ("vin", vin),
            ("page_count", page_count),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                params.append(value)
        params.append(document_id)
        self.db.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
            tuple(params),
        )

    def list_all(self, status: str | None = None, limit: int = 200) -> list[DocumentRecord]:
        if status:
            rows = self.db.query(
                f"SELECT {self._COLUMNS} FROM documents WHERE status = ? "
                "ORDER BY ingested_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self.db.query(
                f"SELECT {self._COLUMNS} FROM documents ORDER BY ingested_at DESC LIMIT ?",
                (limit,),
            )
        return [DocumentRecord(**row) for row in rows]


class ConversationRepository(_BaseRepository):
    """Persist conversations and their messages (used by memory in Phase 7)."""

    def create(self, title: str = "New conversation", vehicle_id: int | None = None) -> Conversation:
        conv_id = self.db.execute(
            "INSERT INTO conversations (title, vehicle_id) VALUES (?, ?)",
            (title, vehicle_id),
        )
        conversation = self.get(conv_id)
        if conversation is None:  # pragma: no cover - defensive
            raise RuntimeError("Failed to create conversation")
        return conversation

    def get(self, conversation_id: int) -> Conversation | None:
        row = self.db.query_one(
            "SELECT id, title, vehicle_id, created_at, updated_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return Conversation(**row) if row else None

    def list_all(self, limit: int = 50) -> list[Conversation]:
        rows = self.db.query(
            "SELECT id, title, vehicle_id, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [Conversation(**row) for row in rows]

    def update_vehicle(self, conversation_id: int, vehicle_id: int) -> None:
        self.db.execute(
            "UPDATE conversations SET vehicle_id = ?, updated_at = ? WHERE id = ?",
            (vehicle_id, utcnow_iso(), conversation_id),
        )

    def clear_messages(self, conversation_id: int) -> None:
        """Remove every message belonging to a conversation."""
        self.db.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )

    def delete(self, conversation_id: int) -> bool:
        """Delete a conversation (and its messages) by id; True when it existed."""
        self.clear_messages(conversation_id)
        return (
            self.db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            > 0
        )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_name: str | None = None,
        citations: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        payload = json.dumps(citations, default=str) if citations else None
        message_id = self.db.execute(
            """
            INSERT INTO messages (conversation_id, role, content, tool_name, citations)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, tool_name, payload),
        )
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (utcnow_iso(), conversation_id),
        )
        return message_id

    def list_messages(self, conversation_id: int, limit: int = 200) -> list[Message]:
        rows = self.db.query(
            """
            SELECT id, conversation_id, role, content, tool_name, citations, created_at
            FROM messages WHERE conversation_id = ? ORDER BY id LIMIT ?
            """,
            (conversation_id, limit),
        )
        return [Message(**row) for row in rows]
