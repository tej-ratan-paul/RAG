"""Document ingestion package.

Loads, cleans, chunks, embeds, and indexes source documents (PDFs, text)
into the vector store while tracking each file in the SQLite ``documents``
table.
"""

from __future__ import annotations

from auto_rag.ingestion.pipeline import IngestionPipeline, IngestResult

__all__ = ["IngestResult", "IngestionPipeline"]
