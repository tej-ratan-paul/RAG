"""Tests for the repository layer."""

from __future__ import annotations

import pytest

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
from auto_rag.db.repositories import (
    ConversationRepository,
    DocumentRepository,
    DTCCodeRepository,
    LaborRepository,
    MaintenanceRepository,
    PartRepository,
    ServiceHistoryRepository,
    VehicleRepository,
)
from auto_rag.errors import DatabaseError


class TestVehicleRepository:
    def test_upsert_inserts_and_reads_back(self, db: Database) -> None:
        repo = VehicleRepository(db)
        vehicle_id = repo.upsert(
            Vehicle(make="Toyota", model="Camry", year=2018, vin="X123", mileage=50000)
        )
        saved = repo.get_by_id(vehicle_id)
        assert saved is not None
        assert saved.display_name == "2018 Toyota Camry"
        assert saved.vin == "X123"

    def test_upsert_updates_existing_by_vin(self, db: Database) -> None:
        repo = VehicleRepository(db)
        first = repo.upsert(Vehicle(make="Toyota", model="Camry", year=2018, vin="X123"))
        second = repo.upsert(
            Vehicle(make="Toyota", model="Camry", year=2018, vin="X123", mileage=90000)
        )
        assert first == second
        assert repo.get_by_id(first).mileage == 90000

    def test_search_and_list(self, db: Database) -> None:
        repo = VehicleRepository(db)
        repo.upsert(Vehicle(make="Toyota", model="Camry", year=2018, vin="A"))
        repo.upsert(Vehicle(make="Honda", model="Civic", year=2016, vin="B"))
        toytas = repo.search(make="Toyota")
        assert len(toytas) == 1 and toytas[0].model == "Camry"
        assert len(repo.list_all()) == 2

    def test_delete(self, db: Database) -> None:
        repo = VehicleRepository(db)
        vid = repo.upsert(Vehicle(make="Toyota", model="Camry", year=2018))
        assert repo.delete(vid) is True
        assert repo.get_by_id(vid) is None
        assert repo.delete(vid) is False


class TestPartRepository:
    def test_upsert_and_stock_update(self, db: Database) -> None:
        repo = PartRepository(db)
        part = Part(part_number="PN-1", name="Oil Filter", quantity_on_hand=5, unit_price=9.99)
        part_id = repo.upsert(part)
        saved = repo.get_by_part_number("pn-1")
        assert saved is not None and saved.id == part_id
        assert saved.quantity_on_hand == 5
        assert saved.in_stock is True

        updated = repo.update_stock("PN-1", -3)
        assert updated is not None
        assert updated.quantity_on_hand == 2

    def test_search_with_compatibility_filters(self, db: Database) -> None:
        repo = PartRepository(db)
        repo.upsert(Part(part_number="P1", name="Brake Pads", compatible_make="Toyota"))
        repo.upsert(Part(part_number="P2", name="Air Filter", compatible_make="Honda"))
        matches = repo.search("brake", make="Toyota")
        assert [p.part_number for p in matches] == ["P1"]
        none = repo.search("brake", make="Honda")
        assert none == []


class TestLaborRepository:
    def test_estimate_cost(self, db: Database) -> None:
        repo = LaborRepository(db)
        repo.upsert(
            LaborOperation(
                operation_code="BR-201",
                description="Brake replacement",
                standard_hours=1.5,
                rate_per_hour=125.00,
            )
        )
        assert repo.estimate_cost("BR-201") == 187.5
        assert repo.estimate_cost("MISSING") is None
        assert repo.search("brake")[0].operation_code == "BR-201"


class TestServiceHistoryRepository:
    def test_add_and_list_by_vehicle(self, db: Database) -> None:
        vehicle_id = VehicleRepository(db).upsert(
            Vehicle(make="Toyota", model="Camry", year=2018)
        )
        repo = ServiceHistoryRepository(db)
        repo.add(
            ServiceRecord(
                vehicle_id=vehicle_id,
                service_date="2026-01-01",
                mileage=10000,
                description="Oil change",
                total_cost=120.0,
            )
        )
        repo.add(
            ServiceRecord(
                vehicle_id=vehicle_id,
                service_date="2026-03-01",
                mileage=12000,
                description="Brakes",
            )
        )
        records = repo.list_by_vehicle(vehicle_id)
        assert len(records) == 2
        assert records[0].description == "Brakes"  # newest first
        assert repo.recent(1)[0].description == "Brakes"


class TestMaintenanceRepository:
    def test_list_for_vehicle(self, db: Database) -> None:
        repo = MaintenanceRepository(db)
        repo.upsert(MaintenanceTask(task="Oil change", interval_miles=7500, interval_months=6))
        repo.upsert(
            MaintenanceTask(
                task="Camry special", make="Toyota", model="Camry", interval_miles=5000
            )
        )
        repo.upsert(
            MaintenanceTask(
                task="Honda special", make="Honda", model="Civic", interval_miles=5000
            )
        )
        vehicle = Vehicle(make="Toyota", model="Camry", year=2018)
        tasks = {t.task for t in repo.list_for_vehicle(vehicle)}
        assert tasks == {"Oil change", "Camry special"}


class TestDTCCodeRepository:
    def test_get_by_code_normalises_case(self, db: Database) -> None:
        repo = DTCCodeRepository(db)
        repo.upsert(DTCCode(code="p0301", description="Cylinder 1 misfire"))
        saved = repo.get_by_code("P0301")
        assert saved is not None
        assert saved.code == "P0301"

    def test_search(self, db: Database) -> None:
        repo = DTCCodeRepository(db)
        repo.upsert(DTCCode(code="P0420", description="Catalyst efficiency"))
        repo.upsert(DTCCode(code="P0171", description="System too lean"))
        assert len(repo.search("catalyst")) == 1
        assert len(repo.search("lean")) == 1


class TestDocumentRepository:
    def test_add_and_update_status(self, db: Database) -> None:
        repo = DocumentRepository(db)
        doc = DocumentRecord(
            source_path="data/documents/manual.pdf",
            file_hash="abc123",
            doc_type="service_manual",
            title="Engine Manual",
        )
        doc_id = repo.add(doc)
        saved = repo.get_by_hash("abc123")
        assert saved is not None and saved.title == "Engine Manual"

        repo.update_status(doc_id, "indexed", chunk_count=42)
        updated = repo.get_by_hash("abc123")
        assert updated.status == "indexed"
        assert updated.chunk_count == 42

        # unique hash prevents duplicate ingestion (pipeline checks first)
        with pytest.raises(DatabaseError):
            repo.add(
                DocumentRecord(
                    source_path="x.pdf", file_hash="abc123", doc_type="tsb", title="dup"
                )
            )
        assert len(repo.list_all()) == 1


class TestConversationRepository:
    def test_create_and_message_flow(self, db: Database) -> None:
        repo = ConversationRepository(db)
        conversation = repo.create(title="Camry diagnostics")
        assert isinstance(conversation, Conversation)
        assert conversation.id is not None

        user_msg = repo.add_message(conversation.id, "user", "P0300 misfire?")
        assistant_msg = repo.add_message(
            conversation.id, "assistant", "Check spark plugs.", tool_name="dtc_search",
            citations=[{"source": "DTC P0300", "page": 1}],
        )
        assert user_msg > 0 and assistant_msg > user_msg

        messages = repo.list_messages(conversation.id)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert isinstance(messages[1], Message)
        assert 'DTC P0300' in messages[1].citations

    def test_update_vehicle(self, db: Database) -> None:
        repo = ConversationRepository(db)
        vehicle_id = VehicleRepository(db).upsert(
            Vehicle(make="Toyota", model="Camry", year=2018)
        )
        conversation = repo.create()
        repo.update_vehicle(conversation.id, vehicle_id)
        assert repo.get(conversation.id).vehicle_id == vehicle_id

    def test_clear_messages(self, db: Database) -> None:
        repo = ConversationRepository(db)
        conversation = repo.create(title="Brakes")
        repo.add_message(conversation.id, "user", "front brake noise?")
        repo.add_message(conversation.id, "assistant", "inspect the pads")
        assert len(repo.list_messages(conversation.id)) == 2

        repo.clear_messages(conversation.id)

        assert repo.list_messages(conversation.id) == []

    def test_delete_conversation(self, db: Database) -> None:
        repo = ConversationRepository(db)
        conversation = repo.create(title="Torque specs")
        repo.add_message(conversation.id, "user", "lug nut torque?")

        assert repo.delete(conversation.id) is True
        assert repo.get(conversation.id) is None
        assert repo.list_messages(conversation.id) == []
        assert repo.delete(conversation.id) is False
