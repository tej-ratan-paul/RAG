"""Application exception hierarchy.

All modules raise specialised subclasses of :class:`AutoRAGError` so callers
(agent, UI, CLI) can handle failure modes predictably.
"""

from __future__ import annotations


class AutoRAGError(Exception):
    """Base class for all application-specific errors."""


class ConfigurationError(AutoRAGError):
    """Raised when the application settings are invalid or incomplete."""


class DatabaseError(AutoRAGError):
    """Raised for SQLite/schema failures."""


class IngestionError(AutoRAGError):
    """Raised during document loading, cleaning, or chunking."""


class VectorStoreError(AutoRAGError):
    """Raised for vector store creation, indexing, or query failures."""


class RetrievalError(AutoRAGError):
    """Raised when retrieval (dense/hybrid/rerank) fails."""


class AgentError(AutoRAGError):
    """Raised for agent orchestration failures."""


class LLMError(AutoRAGError):
    """Raised when the underlying LLM cannot be reached or returns errors."""
