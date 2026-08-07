"""AutoRAG Repair Assistant.

An agentic RAG system that answers automobile repair questions using OEM
service manuals, DTC codes, TSBs, wiring diagrams, and structured workshop
data (parts, labor rates, service history, maintenance schedules).
"""

from __future__ import annotations

from auto_rag.constants import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION", "__version__"]

__version__: str = APP_VERSION
