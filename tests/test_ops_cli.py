"""Tests for the auto-rag-health and auto-rag-config-check CLIs."""

from __future__ import annotations

from pathlib import Path

import pytest

import auto_rag.ops.health_cli as health_cli
import auto_rag.ops.validate as validate
from auto_rag.config import Settings
from auto_rag.ops.checks import CheckResult


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        paths={
            "project_root": tmp_path,
            "data_dir": tmp_path / "data",
            "documents_dir": tmp_path / "data" / "documents",
            "db_dir": tmp_path / "data" / "db",
            "logs_dir": tmp_path / "data" / "logs",
        },
    )


def _all_ok_results() -> list[CheckResult]:
    return [
        CheckResult("settings", True, "ok"),
        CheckResult("directories", True, "ok"),
        CheckResult("database", True, "ok"),
        CheckResult("vector_store", True, "ok"),
        CheckResult("llm", True, "ok"),
    ]


def test_health_cli_renders_text(tmp_path: Path) -> None:
    rendered = health_cli._render(_all_ok_results(), "text")
    assert "[OK  ] settings" in rendered
    assert "ms)" in rendered


def test_health_cli_renders_json(tmp_path: Path) -> None:
    import json

    payload = json.loads(health_cli._render(_all_ok_results(), "json"))
    assert payload[0]["name"] == "settings"
    assert payload[0]["ok"] is True


def test_health_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(health_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(health_cli, "run_checks", lambda *a, **k: _all_ok_results())
    assert health_cli.main([]) == 0

    failing = [CheckResult("database", False, "boom")]
    monkeypatch.setattr(health_cli, "run_checks", lambda *a, **k: failing)
    assert health_cli.main(["--format", "json"]) == 1


def test_health_cli_serve_uses_settings_health_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[_FakeHealthServer] = []

    class _FakeHealthServer:
        def __init__(self, settings, *, host, port, deep, include_llm):
            self.host = host
            self.port = port
            self.deep = deep
            self.include_llm = include_llm
            instances.append(self)

        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            pass

    settings = _make_settings(tmp_path)
    settings.health.host = "10.0.0.5"
    settings.health.port = 9111
    monkeypatch.setattr(health_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(health_cli, "HealthServer", _FakeHealthServer)

    assert health_cli.main(["--serve", "--no-llm"]) == 0
    assert instances[-1].host == "10.0.0.5"
    assert instances[-1].port == 9111
    assert instances[-1].include_llm is False

    assert health_cli.main(["--serve", "--host", "0.0.0.0", "--port", "9090"]) == 0
    assert instances[-1].host == "0.0.0.0"
    assert instances[-1].port == 9090
    assert instances[-1].include_llm is True


def test_validate_sections_ok(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    settings.prepare_directories()
    items = validate.validate_sections(settings)
    assert all(item.ok for item in items)
    assert {item.section for item in items} >= {
        "app",
        "llm.provider",
        "vectorstore.backend",
        "embeddings.device",
    }


def test_validate_reports_unsupported_provider(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path).model_copy(deep=True)
    settings.llm.provider = "watson"
    items = validate.validate_sections(settings)
    llm_item = next(item for item in items if item.section == "llm.provider")
    assert not llm_item.ok


def test_validate_run_validation_includes_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path)
    settings.prepare_directories()

    def fake_run_checks(settings, *, include_llm):
        return [
            CheckResult("database", True, "ok"),
            CheckResult("vector_store", True, "ok"),
            CheckResult("llm", True, "ok") if include_llm else CheckResult("settings", True, "ok"),
        ]

    monkeypatch.setattr(validate, "run_checks", fake_run_checks)
    items = validate.run_validation(settings, include_llm=True)
    sections = {item.section for item in items}
    assert {"database", "vector_store", "llm"} <= sections
    assert all(item.ok for item in items)
