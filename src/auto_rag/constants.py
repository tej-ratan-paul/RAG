"""Central application constants.

These values are shared across modules and feed the default configuration.
Keeping them here avoids magic strings scattered through the codebase.
"""

from __future__ import annotations

from typing import Final

APP_NAME: Final[str] = "AutoRAG Repair Assistant"
APP_VERSION: Final[str] = "0.1.0"

# Embeddings
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: Final[int] = 384

# LLM
DEFAULT_LLM_PROVIDER: Final[str] = "ollama"
DEFAULT_LLM_MODEL: Final[str] = "llama3.1"

# Vector store
DEFAULT_VECTORSTORE_BACKEND: Final[str] = "chroma"
DEFAULT_COLLECTION_NAME: Final[str] = "auto_rag_documents"

# Persistence
SQLITE_DB_FILENAME: Final[str] = "auto_rag.db"
CHROMA_DIR_NAME: Final[str] = "chroma"

# Document taxonomy
DOCUMENT_TYPE_SERVICE_MANUAL: Final[str] = "service_manual"
DOCUMENT_TYPE_REPAIR_MANUAL: Final[str] = "repair_manual"
DOCUMENT_TYPE_DTC: Final[str] = "dtc"
DOCUMENT_TYPE_TSB: Final[str] = "tsb"
DOCUMENT_TYPE_WIRING_DIAGRAM: Final[str] = "wiring_diagram"

DOCUMENT_TYPES: Final[tuple[str, ...]] = (
    DOCUMENT_TYPE_SERVICE_MANUAL,
    DOCUMENT_TYPE_REPAIR_MANUAL,
    DOCUMENT_TYPE_DTC,
    DOCUMENT_TYPE_TSB,
    DOCUMENT_TYPE_WIRING_DIAGRAM,
)
