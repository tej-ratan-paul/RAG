"""Tests for the demo data seeder."""

from __future__ import annotations

from auto_rag.db.connection import Database
from auto_rag.db.seeder import seed_demo_data


def test_seed_is_idempotent(db: Database) -> None:
    first = seed_demo_data(db)
    second = seed_demo_data(db)

    assert first["vehicles"] > 0
    assert second["vehicles"] == 0  # already seeded
    assert second["parts"] == 0
    assert second["dtc_codes"] == 0
    assert second["maintenance"] == 0

    for table in ("vehicles", "parts", "labor_operations", "service_history",
                  "maintenance_schedule", "dtc_codes"):
        count = db.scalar(f"SELECT COUNT(*) FROM {table}")
        assert count > 0


def test_seeded_data_is_queryable(db: Database) -> None:
    seed_demo_data(db)
    assert db.scalar("SELECT COUNT(*) FROM vehicles") == 2
    assert db.scalar("SELECT COUNT(*) FROM dtc_codes") >= 8
    camry = db.query_one("SELECT * FROM vehicles WHERE make = 'Toyota'")
    assert camry is not None and camry["vin"].startswith("4T1")
    water_pump = db.query_one("SELECT * FROM parts WHERE part_number = '16400-39745'")
    assert water_pump["quantity_on_hand"] == 0
