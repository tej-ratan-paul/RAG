"""Pydantic domain models for the structured data layer.

These models are the boundary objects between repositories and the rest of
the application (agent tools, UI). Rows are mapped to/from these types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal[
    "service_manual",
    "repair_manual",
    "dtc",
    "tsb",
    "wiring_diagram",
]

DocStatus = Literal["pending", "indexed", "failed"]

MessageRole = Literal["user", "assistant", "system", "tool"]


class Vehicle(BaseModel):
    """A customer vehicle under service."""

    id: int | None = None
    make: str
    model: str
    year: int = Field(ge=1900, le=2100)
    engine: str | None = None
    vin: str | None = None
    mileage: int | None = Field(default=None, ge=0)
    notes: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def display_name(self) -> str:
        """Human-readable vehicle label, e.g. '2018 Toyota Camry'."""
        return f"{self.year} {self.make} {self.model}"


class Part(BaseModel):
    """A spare part in inventory."""

    id: int | None = None
    part_number: str
    name: str
    description: str | None = None
    category: str | None = None
    manufacturer: str | None = None
    compatible_make: str | None = None
    compatible_model: str | None = None
    compatible_years: str | None = None
    quantity_on_hand: int = Field(default=0, ge=0)
    reorder_level: int = Field(default=0, ge=0)
    unit_price: float = Field(default=0.0, ge=0.0)
    location: str | None = None
    updated_at: str | None = None

    @property
    def in_stock(self) -> bool:
        """True when stock exceeds the reorder level."""
        return self.quantity_on_hand > self.reorder_level


class LaborOperation(BaseModel):
    """A standard repair operation with book time and shop rate."""

    id: int | None = None
    operation_code: str
    description: str
    skill_level: str = "Technician"
    standard_hours: float = Field(gt=0.0)
    rate_per_hour: float = Field(ge=0.0)
    notes: str | None = None
    updated_at: str | None = None

    @property
    def estimated_cost(self) -> float:
        """Standard hours multiplied by the hourly rate."""
        return round(self.standard_hours * self.rate_per_hour, 2)


class ServiceRecord(BaseModel):
    """A past service event recorded against a vehicle."""

    id: int | None = None
    vehicle_id: int
    service_date: str
    mileage: int | None = None
    description: str
    dtc_codes: str | None = None
    parts_used: str | None = None
    labor_operation_code: str | None = None
    total_cost: float | None = None
    technician: str | None = None
    notes: str | None = None
    created_at: str | None = None


class MaintenanceTask(BaseModel):
    """A scheduled maintenance task with service intervals."""

    id: int | None = None
    task: str
    description: str | None = None
    make: str | None = None
    model: str | None = None
    year: int | None = None
    interval_miles: int | None = Field(default=None, gt=0)
    interval_months: int | None = Field(default=None, gt=0)
    parts_required: str | None = None
    priority: str = "standard"
    updated_at: str | None = None


class DTCCode(BaseModel):
    """A diagnostic trouble code definition."""

    id: int | None = None
    code: str
    description: str
    system: str | None = None
    manufacturer: str | None = None
    possible_causes: str | None = None
    diagnostic_notes: str | None = None
    severity: str = "unknown"
    updated_at: str | None = None


class DocumentRecord(BaseModel):
    """A source file tracked through the ingestion pipeline."""

    id: int | None = None
    source_path: str
    file_hash: str
    doc_type: DocType
    title: str | None = None
    make: str | None = None
    model: str | None = None
    year: int | None = None
    engine: str | None = None
    vin: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    status: DocStatus = "pending"
    error: str | None = None
    ingested_at: str | None = None
    updated_at: str | None = None


class Conversation(BaseModel):
    """A persisted chat conversation, optionally linked to a vehicle."""

    id: int | None = None
    title: str = "New conversation"
    vehicle_id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class Message(BaseModel):
    """A single message inside a conversation."""

    id: int | None = None
    conversation_id: int
    role: MessageRole
    content: str
    tool_name: str | None = None
    citations: str | None = None
    created_at: str | None = None
