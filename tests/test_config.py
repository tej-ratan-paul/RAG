"""Tests for configuration loading, validation, and derived paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_rag.config import (
    LoggingConfig,
    Settings,
    get_settings,
    invalidate_settings_cache,
)
from auto_rag.errors import ConfigurationError
from auto_rag.utils.paths import find_project_root, project_root


def test_find_project_root_detects_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()
    assert project_root() == root


def test_defaults_are_sane(settings: Settings) -> None:
    assert settings.app.name == "AutoRAG Repair Assistant"
    assert settings.app.environment == "development"
    assert settings.embeddings.dimension == 384
    assert settings.chunking.overlap < settings.chunking.size
    assert settings.sqlite_path.parent == settings.paths.db_dir
    assert settings.sqlite_path.name == "auto_rag.db"
    assert settings.chroma_persist_dir.parent == settings.paths.db_dir
    assert settings.log_file_path.parent == settings.paths.logs_dir
    assert settings.health.host == "0.0.0.0"
    assert settings.health.port == 8080


def test_paths_resolve_under_data_dir(settings: Settings) -> None:
    assert settings.paths.data_dir == settings.paths.project_root / "data"
    assert settings.paths.documents_dir == settings.paths.data_dir / "documents"
    assert settings.paths.db_dir == settings.paths.data_dir / "db"
    assert settings.paths.logs_dir == settings.paths.data_dir / "logs"


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__NAME", "Test Shop")
    monkeypatch.setenv("APP__ENVIRONMENT", "staging")
    monkeypatch.setenv("RETRIEVAL__TOP_K", "7")
    monkeypatch.setenv("CHUNKING__SIZE", "1000")
    monkeypatch.setenv("CHUNKING__OVERLAP", "100")

    override = Settings(_env_file=None)
    assert override.app.name == "Test Shop"
    assert override.app.environment == "staging"
    assert override.retrieval.top_k == 7
    assert override.chunking.size == 1000
    assert override.chunking.overlap == 100


def test_health_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH__HOST", "127.0.0.1")
    monkeypatch.setenv("HEALTH__PORT", "9090")

    override = Settings(_env_file=None)
    assert override.health.host == "127.0.0.1"
    assert override.health.port == 9090


def test_invalid_health_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH__PORT", "70000")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGING__LEVEL", "NOISY")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__ENVIRONMENT", "experimental")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunking_overlap_must_be_smaller_than_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKING__SIZE", "100")
    monkeypatch.setenv("CHUNKING__OVERLAP", "150")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_retrieval_ordering_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL__TOP_K", "10")
    monkeypatch.setenv("RETRIEVAL__HYBRID_TOP_K", "5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_explicit_sqlite_path_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = (tmp_path / "custom.db").expanduser().resolve()
    monkeypatch.setenv("DATABASE__PATH", str(expected))
    settings = Settings(_env_file=None)
    assert settings.sqlite_path == expected


def test_get_settings_returns_cached_singleton() -> None:
    invalidate_settings_cache()
    first = get_settings()
    second = get_settings()
    assert first is second
    invalidate_settings_cache()


def test_load_failure_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import auto_rag.config as config_module

    monkeypatch.setenv("LOGGING__LEVEL", "BOGUS")

    def bad_load() -> Settings:
        raise ValidationError.from_exception_data("Settings", [])

    original_load = config_module.Settings.load
    config_module.Settings.load = bad_load  # type: ignore[method-assign]
    try:
        with pytest.raises(ConfigurationError):
            config_module.get_settings()
    finally:
        config_module.Settings.load = original_load  # type: ignore[method-assign]
        invalidate_settings_cache()


def test_logging_config_levels() -> None:
    assert LoggingConfig(level="debug").level == "DEBUG"
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")
