"""Application configuration.

A single, validated :class:`Settings` object is the source of truth for the
whole application. Values come from (highest priority first):

1. Environment variables / ``.env`` file (``SECTION__FIELD`` naming).
2. Sensible defaults derived from the project root.

Use :func:`get_settings` to obtain the shared singleton; pass ``Settings``
instances around for dependency injection in tests.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from auto_rag.constants import (
    APP_NAME,
    APP_VERSION,
    CHROMA_DIR_NAME,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_VECTORSTORE_BACKEND,
    EMBEDDING_DIMENSION,
    SQLITE_DB_FILENAME,
)
from auto_rag.errors import ConfigurationError
from auto_rag.utils.paths import project_root

__all__ = ["Settings", "get_settings", "LoggingConfig"]

_VALID_LOG_LEVELS: Final[tuple[str, ...]] = (
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
)


class AppConfig(BaseModel):
    """Top-level application metadata."""

    name: str = Field(default=APP_NAME, description="Application display name.")
    version: str = Field(default=APP_VERSION, description="Semantic version.")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Runtime environment."
    )
    debug: bool = Field(default=False, description="Enable debug behaviour.")

    @field_validator("environment")
    @classmethod
    def _normalise_environment(cls, value: str) -> str:
        return value.lower().strip()


class PathsConfig(BaseModel):
    """Filesystem locations. Relative values are resolved against the root."""

    project_root: Path = Field(default_factory=project_root)
    data_dir: Path | None = Field(default=None)
    documents_dir: Path | None = Field(default=None)
    db_dir: Path | None = Field(default=None)
    logs_dir: Path | None = Field(default=None)

    @model_validator(mode="after")
    def _fill_derived_paths(self) -> Self:
        root = self.project_root
        data = (self.data_dir or root / "data").expanduser().resolve()
        self.data_dir = data
        self.documents_dir = (
            self.documents_dir or data / "documents"
        ).expanduser().resolve()
        self.db_dir = (self.db_dir or data / "db").expanduser().resolve()
        self.logs_dir = (self.logs_dir or data / "logs").expanduser().resolve()
        return self


class DatabaseConfig(BaseModel):
    """SQLite connection settings."""

    path: Path | None = Field(default=None, description="Explicit DB file path override.")
    filename: str = Field(default=SQLITE_DB_FILENAME, description="DB filename.")
    timeout: float = Field(default=10.0, ge=1.0, description="Connection timeout (s).")
    journal_mode: Literal["delete", "truncate", "persist", "memory", "wal", "off"] = Field(
        default="wal", description="SQLite journal mode."
    )


class VectorStoreConfig(BaseModel):
    """Vector store backend configuration."""

    backend: Literal["chroma", "qdrant"] = Field(
        default=DEFAULT_VECTORSTORE_BACKEND, description="Vector DB backend."
    )
    persist_dir: Path | None = Field(
        default=None, description="Explicit Chroma persistence dir override."
    )
    collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME, description="Vector collection name."
    )
    distance: Literal["cosine", "l2", "ip"] = Field(
        default="cosine", description="Similarity metric."
    )


class EmbeddingsConfig(BaseModel):
    """HuggingFace sentence-transformers settings."""

    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="HF model name.")
    dimension: int = Field(
        default=EMBEDDING_DIMENSION, ge=64, le=4096, description="Embedding dimension."
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Inference device."
    )
    batch_size: int = Field(default=32, ge=1, description="Encode batch size.")
    normalize: bool = Field(default=True, description="L2-normalise embeddings.")
    cache_enabled: bool = Field(
        default=False,
        description="Persist embeddings on disk keyed by SHA-256 of the text.",
    )
    cache_file: str = Field(
        default="embedding_cache.json",
        description="Embedding cache filename under ``<data_dir>/cache``.",
    )


class LLMConfig(BaseModel):
    """LLM settings (provider defaults to Ollama)."""

    provider: str = Field(default=DEFAULT_LLM_PROVIDER, description="LLM provider.")
    base_url: str = Field(default="http://localhost:11434", description="API host.")
    model: str = Field(default=DEFAULT_LLM_MODEL, description="Model tag.")
    api_key: str = Field(default="", description="API key (OpenAI-compatible only).")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = Field(default=120.0, ge=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    """Retrieval pipeline settings."""

    top_k: int = Field(default=5, ge=1, description="Final number of results.")
    hybrid_search: bool = Field(default=True, description="Combine dense + BM25.")
    hybrid_top_k: int = Field(default=8, ge=1, description="Pre-fusion result count.")
    mmr: bool = Field(default=True, description="Apply Maximal Marginal Relevance.")
    mmr_lambda_mult: float = Field(default=0.7, ge=0.0, le=1.0)
    mmr_fetch_k: int = Field(default=20, ge=1, description="MMR candidate pool size.")
    rerank: bool = Field(default=True, description="Apply cross-encoder reranking.")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model.",
    )
    rerank_top_k: int = Field(default=3, ge=1, description="Post-rerank result count.")

    @model_validator(mode="after")
    def _validate_ordering(self) -> Self:
        if self.hybrid_top_k < self.top_k:
            raise ValueError("RETRIEVAL__HYBRID_TOP_K must be >= RETRIEVAL__TOP_K")
        if self.mmr_fetch_k < self.top_k:
            raise ValueError("RETRIEVAL__MMR_FETCH_K must be >= RETRIEVAL__TOP_K")
        if self.rerank_top_k > self.top_k:
            raise ValueError("RETRIEVAL__RERANK_TOP_K must be <= RETRIEVAL__TOP_K")
        return self


class ChunkingConfig(BaseModel):
    """Document chunking settings."""

    size: int = Field(default=800, ge=64, description="Target chunk size (chars).")
    overlap: int = Field(default=160, ge=0, description="Chunk overlap (chars).")

    @model_validator(mode="after")
    def _validate_overlap(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("CHUNKING__OVERLAP must be smaller than CHUNKING__SIZE")
        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Root log level.")
    json_format: bool = Field(default=False, description="Emit structured JSON records.")
    console: bool = Field(default=True, description="Log to stdout.")
    file: bool = Field(default=True, description="Log to a rotating file.")
    file_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1_000)
    file_backup_count: int = Field(default=3, ge=0)

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        level = value.upper().strip()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOGGING__LEVEL must be one of {_VALID_LOG_LEVELS}, got {value!r}"
            )
        return level


class HealthConfig(BaseModel):
    """HTTP health probe server settings."""

    host: str = Field(default="0.0.0.0", description="Bind address.")
    port: int = Field(default=8080, ge=1, le=65535, description="Listen port.")


class Settings(BaseSettings):
    """Validated application configuration (see module docstring)."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)

    # ------------------------------------------------------------------ #
    # Derived locations
    # ------------------------------------------------------------------ #
    @property
    def sqlite_path(self) -> Path:
        """Absolute path to the SQLite database file."""
        if self.database.path is not None:
            return self.database.path.expanduser().resolve()
        return self.paths.db_dir / self.database.filename

    @property
    def chroma_persist_dir(self) -> Path:
        """Absolute Chroma persistence directory."""
        if self.vectorstore.persist_dir is not None:
            return self.vectorstore.persist_dir.expanduser().resolve()
        return self.paths.db_dir / CHROMA_DIR_NAME

    @property
    def log_file_path(self) -> Path:
        """Absolute path of the rotating log file."""
        return self.paths.logs_dir / "auto_rag.log"

    @property
    def embedding_cache_path(self) -> Path:
        """Absolute path of the persistent embedding cache file."""
        return self.paths.data_dir / "cache" / self.embeddings.cache_file

    @classmethod
    def load(cls) -> Settings:
        """Load settings from the repository-root ``.env`` plus env vars."""
        env_file = project_root() / ".env"
        return cls(_env_file=env_file)

    def prepare_directories(self) -> None:
        """Create all runtime directories required by the application."""
        from auto_rag.utils.paths import ensure_directory

        for directory in (
            self.paths.data_dir,
            self.paths.documents_dir,
            self.paths.db_dir,
            self.paths.logs_dir,
        ):
            ensure_directory(directory)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application-wide :class:`Settings` singleton."""
    try:
        settings = Settings.load()
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        raise ConfigurationError(f"Failed to load settings: {exc}") from exc
    logging.getLogger(__name__).debug("Settings loaded from %s", settings)
    return settings


def invalidate_settings_cache() -> None:
    """Clear the cached settings singleton (used by tests)."""
    get_settings.cache_clear()
