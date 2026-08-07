"""Demo data seeder.

Populates an empty database with realistic workshop data so the agent and UI
can be exercised end-to-end without real customer records. Seeding is
idempotent: each table is only populated when empty.
"""

from __future__ import annotations

import logging

from auto_rag.db.connection import Database
from auto_rag.db.models import (
    DTCCode,
    LaborOperation,
    MaintenanceTask,
    Part,
    ServiceRecord,
    Vehicle,
)
from auto_rag.db.repositories import (
    DTCCodeRepository,
    LaborRepository,
    MaintenanceRepository,
    PartRepository,
    ServiceHistoryRepository,
    VehicleRepository,
)

logger = logging.getLogger(__name__)

_VEHICLES = [
    Vehicle(
        make="Toyota",
        model="Camry",
        year=2018,
        engine="2.5L I4",
        vin="4T1B11HK3JU123456",
        mileage=84500,
    ),
    Vehicle(
        make="Honda",
        model="Civic",
        year=2016,
        engine="1.5L I4 Turbo",
        vin="2HGFC2F59GH654321",
        mileage=102300,
    ),
]

_PARTS = [
    Part(part_number="90915-YZZE1", name="Engine Oil Filter", description="OEM oil filter for 2.5L engine", category="Filters", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=25, reorder_level=10, unit_price=8.99, location="A1-02"),
    Part(part_number="15400-PLM-A02", name="Engine Oil Filter", description="OEM oil filter for 1.5L turbo", category="Filters", manufacturer="Honda", compatible_make="Honda", compatible_model="Civic", compatible_years="2016-2021", quantity_on_hand=18, reorder_level=10, unit_price=7.49, location="A1-03"),
    Part(part_number="04465-06360", name="Front Brake Pad Set", description="Front disc brake pad set, semi-metallic", category="Brakes", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=6, reorder_level=4, unit_price=64.50, location="B2-11"),
    Part(part_number="27060-0Y080", name="Alternator", description="120A alternator assembly", category="Electrical", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=2, reorder_level=2, unit_price=318.00, location="C3-07"),
    Part(part_number="16400-39745", name="Water Pump", description="Engine coolant water pump", category="Cooling", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=0, reorder_level=2, unit_price=92.40, location="C1-09"),
    Part(part_number="90919-01231", name="Spark Plug (IR)", description="Iridium spark plug, qty 4 required", category="Ignition", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=40, reorder_level=16, unit_price=14.25, location="D1-04"),
    Part(part_number="22270-38010", name="MAF Sensor", description="Mass air flow sensor assembly", category="Sensors", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=3, reorder_level=2, unit_price=136.00, location="E2-05"),
    Part(part_number="89465-0E050", name="O2 Sensor (Front)", description="Air-fuel ratio sensor, bank 1", category="Sensors", manufacturer="Toyota", compatible_make="Toyota", compatible_model="Camry", compatible_years="2018-2024", quantity_on_hand=1, reorder_level=2, unit_price=164.80, location="E2-06"),
]

_LABOR = [
    LaborOperation(operation_code="OP-101", description="Engine oil and filter change", skill_level="Technician", standard_hours=0.5, rate_per_hour=125.00),
    LaborOperation(operation_code="BR-201", description="Front brake pad replacement (both sides)", skill_level="Technician", standard_hours=1.5, rate_per_hour=125.00),
    LaborOperation(operation_code="EL-301", description="Alternator replacement", skill_level="Specialist", standard_hours=2.0, rate_per_hour=145.00),
    LaborOperation(operation_code="CO-401", description="Coolant flush and refill", skill_level="Technician", standard_hours=1.0, rate_per_hour=125.00),
    LaborOperation(operation_code="EN-501", description="Spark plug replacement (4-cyl)", skill_level="Technician", standard_hours=1.0, rate_per_hour=125.00),
    LaborOperation(operation_code="SE-601", description="Diagnostic scan and DTC readout", skill_level="Master", standard_hours=0.7, rate_per_hour=150.00),
]

_DTCS = [
    DTCCode(code="P0301", description="Cylinder 1 misfire detected", system="Powertrain", manufacturer="All", possible_causes="Faulty spark plug, ignition coil, fuel injector, low compression, vacuum leak", diagnostic_notes="Check stored misfire counter; swap coil to another cylinder to isolate.", severity="high"),
    DTCCode(code="P0300", description="Random/multiple cylinder misfire detected", system="Powertrain", manufacturer="All", possible_causes="Vacuum leak, fuel delivery issue, ignition system, EGR system, low fuel pressure", diagnostic_notes="Inspect intake for leaks; verify fuel trims and spark condition.", severity="high"),
    DTCCode(code="P0420", description="Catalyst system efficiency below threshold (Bank 1)", system="Powertrain", manufacturer="All", possible_causes="Faulty catalytic converter, O2 sensor deterioration, exhaust leak", diagnostic_notes="Compare downstream O2 switching; verify converter temperature.", severity="medium"),
    DTCCode(code="P0171", description="System too lean (Bank 1)", system="Powertrain", manufacturer="All", possible_causes="Mass air flow sensor, vacuum leak, fuel pressure, MAF contamination", diagnostic_notes="Check MAF readings and fuel trims at idle and cruise.", severity="medium"),
    DTCCode(code="P0442", description="EVAP system small leak detected", system="Powertrain", manufacturer="All", possible_causes="Loose gas cap, EVAP hose leak, purge valve, canister", diagnostic_notes="Perform smoke test on EVAP system.", severity="low"),
    DTCCode(code="P0013", description="'B' camshaft position - actuator circuit open (Bank 1)", system="Powertrain", manufacturer="All", possible_causes="VVT solenoid circuit, failed solenoid, wiring damage, low oil pressure", diagnostic_notes="Check oil level and VVT solenoid resistance.", severity="medium"),
    DTCCode(code="U0100", description="Lost communication with ECM/PCM", system="Network", manufacturer="All", possible_causes="Faulty module, CAN bus wiring, terminal corrosion, low battery voltage", diagnostic_notes="Verify module power/ground and CAN resistance.", severity="critical"),
    DTCCode(code="C1201", description="Engine control system malfunction (ABS/VSC)", system="Chassis", manufacturer="Toyota", possible_causes="Engine control module communication, wheel speed sensor, wiring", diagnostic_notes="Clear codes and verify with ABS/VSC monitor.", severity="medium"),
]

_MAINTENANCE = [
    MaintenanceTask(task="Engine oil and filter", description="Replace engine oil and filter", make=None, model=None, interval_miles=7500, interval_months=6, parts_required="Oil, Oil Filter"),
    MaintenanceTask(task="Tire rotation", description="Rotate tires front to rear", make=None, model=None, interval_miles=7500, interval_months=6),
    MaintenanceTask(task="Brake inspection", description="Inspect brake pads, rotors, fluid", make=None, model=None, interval_miles=15000, interval_months=12),
    MaintenanceTask(task="Cabin air filter", description="Replace cabin air filter", make=None, model=None, interval_miles=15000, interval_months=12, parts_required="Cabin Air Filter"),
    MaintenanceTask(task="Engine air filter", description="Replace engine air filter", make=None, model=None, interval_miles=30000, interval_months=24, parts_required="Air Filter"),
    MaintenanceTask(task="Coolant flush", description="Flush and replace engine coolant", make=None, model=None, interval_miles=60000, interval_months=60, parts_required="Coolant"),
    MaintenanceTask(task="Transmission fluid service", description="Drain and refill automatic transmission fluid", make=None, model=None, interval_miles=60000, interval_months=48, parts_required="ATF"),
    MaintenanceTask(task="Spark plugs", description="Replace spark plugs (4-cyl)", make=None, model=None, interval_miles=100000, interval_months=120, parts_required="Spark Plugs x4"),
]

_SERVICE_HISTORY = [
    ServiceRecord(vehicle_id=1, service_date="2026-01-15", mileage=82000, description="Engine oil and filter change; tire rotation", parts_used="90915-YZZE1", labor_operation_code="OP-101", total_cost=150.00, technician="D. Rivera"),
    ServiceRecord(vehicle_id=1, service_date="2026-03-02", mileage=83500, description="Diagnostic scan; replaced front brake pads", dtc_codes="P0300", parts_used="04465-06360", labor_operation_code="BR-201", total_cost=420.00, technician="M. Chen"),
    ServiceRecord(vehicle_id=1, service_date="2026-06-10", mileage=84500, description="Coolant flush and brake inspection", labor_operation_code="CO-401", total_cost=310.00, technician="D. Rivera"),
    ServiceRecord(vehicle_id=2, service_date="2026-02-20", mileage=99000, description="Spark plug replacement and air filter", parts_used="Spark Plugs", labor_operation_code="EN-501", total_cost=290.00, technician="M. Chen"),
    ServiceRecord(vehicle_id=2, service_date="2026-05-05", mileage=102300, description="Diagnostic scan - random misfire, oil and filter change", dtc_codes="P0300", labor_operation_code="SE-601", total_cost=210.00, technician="A. Brooks"),
]


def seed_demo_data(db: Database) -> dict[str, int]:
    """Populate empty tables with demo data; returns counts inserted.

    Each table is only seeded when it is completely empty, making this
    safe to call repeatedly.
    """
    counts: dict[str, int] = {}
    vehicles = VehicleRepository(db).list_all()
    seeded_vehicles = not vehicles
    if seeded_vehicles:
        vehicles = []
        for vehicle in _VEHICLES:
            vehicle_id = VehicleRepository(db).upsert(vehicle)
            vehicles.append(VehicleRepository(db).get_by_id(vehicle_id))
    counts["vehicles"] = len(vehicles) if seeded_vehicles else 0

    counts["parts"] = _seed_parts(db)
    counts["labor_operations"] = _seed_labor(db)
    counts["dtc_codes"] = _seed_dtcs(db)
    counts["maintenance"] = _seed_maintenance(db)
    counts["service_history"] = _seed_service_history(db)
    logger.info("Seeded demo data: %s", counts)
    return counts


def _seed_parts(db: Database) -> int:
    if db.scalar("SELECT COUNT(*) FROM parts") > 0:
        return 0
    for part in _PARTS:
        PartRepository(db).upsert(part)
    return len(_PARTS)


def _seed_labor(db: Database) -> int:
    if db.scalar("SELECT COUNT(*) FROM labor_operations") > 0:
        return 0
    for op in _LABOR:
        LaborRepository(db).upsert(op)
    return len(_LABOR)


def _seed_dtcs(db: Database) -> int:
    if db.scalar("SELECT COUNT(*) FROM dtc_codes") > 0:
        return 0
    for dtc in _DTCS:
        DTCCodeRepository(db).upsert(dtc)
    return len(_DTCS)


def _seed_maintenance(db: Database) -> int:
    if db.scalar("SELECT COUNT(*) FROM maintenance_schedule") > 0:
        return 0
    for task in _MAINTENANCE:
        MaintenanceRepository(db).upsert(task)
    return len(_MAINTENANCE)


def _seed_service_history(db: Database) -> int:
    if db.scalar("SELECT COUNT(*) FROM service_history") > 0:
        return 0
    for record in _SERVICE_HISTORY:
        ServiceHistoryRepository(db).add(record)
    return len(_SERVICE_HISTORY)
