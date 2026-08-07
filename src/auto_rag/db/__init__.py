"""Persistence layer: SQLite connection, models, repositories, and seeding."""

from __future__ import annotations

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
from auto_rag.db.seeder import seed_demo_data

__all__ = [
    "Conversation",
    "ConversationRepository",
    "DTCCode",
    "DTCCodeRepository",
    "Database",
    "DocumentRecord",
    "DocumentRepository",
    "LaborOperation",
    "LaborRepository",
    "MaintenanceRepository",
    "MaintenanceTask",
    "Message",
    "Part",
    "PartRepository",
    "ServiceHistoryRepository",
    "ServiceRecord",
    "Vehicle",
    "VehicleRepository",
    "seed_demo_data",
]
