"""Tests for structured logging setup and output formats."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from auto_rag import logging_config as logging_module
from auto_rag.config import LoggingConfig, Settings
from auto_rag.logging_config import JsonFormatter, get_logger, log_with_fields, setup_logging


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Wipe the root logger handlers and reset the module guard."""
    _clear_root_handlers()
    logging_module._configured = False
    yield
    _clear_root_handlers()
    logging_module._configured = False


def _clear_root_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _with_logging(settings: Settings, **changes) -> Settings:
    cfg = LoggingConfig(**changes)
    return settings.model_copy(update={"logging": cfg})


def test_json_formatter_output() -> None:
    record = logging.LogRecord(
        name="auto_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    formatter = JsonFormatter()
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "auto_rag.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_setup_logging_is_idempotent(settings: Settings) -> None:
    patched = _with_logging(settings, level="DEBUG", console=True, file=False)
    setup_logging(patched)
    handler_count = len(logging.getLogger().handlers)
    setup_logging(patched)
    assert len(logging.getLogger().handlers) == handler_count


def test_logging_writes_to_file(settings: Settings, tmp_path: Path) -> None:
    paths = settings.paths.model_copy(update={"logs_dir": tmp_path})
    patched = _with_logging(
        settings,
        level="INFO",
        console=False,
        file=True,
        json_format=True,
    ).model_copy(update={"paths": paths})

    setup_logging(patched)
    get_logger("auto_rag.test_file").info("file log line")

    log_file = tmp_path / "auto_rag.log"
    assert log_file.is_file()
    last_line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(last_line)
    assert record["message"] == "file log line"


def test_console_handler_emits_to_stdout(settings: Settings, monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr("sys.stdout", buffer)

    patched = _with_logging(settings, level="DEBUG", console=True, file=False, json_format=True)
    setup_logging(patched)

    get_logger("auto_rag.test_console").warning("console says hi")
    assert "console says hi" in buffer.getvalue()


def test_json_formatter_includes_service() -> None:
    record = logging.LogRecord(
        name="auto_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    formatter = JsonFormatter(service="AutoRAG Repair Assistant v0.1.0")
    payload = json.loads(formatter.format(record))
    assert payload["service"] == "AutoRAG Repair Assistant v0.1.0"


def test_json_formatter_includes_fields() -> None:
    record = logging.LogRecord(
        name="auto_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="retrieved",
        args=(),
        exc_info=None,
    )
    record.fields = {"top_k": 5, "latency_ms": 12.3}  # type: ignore[attr-defined]
    payload = json.loads(JsonFormatter().format(record))
    assert payload["top_k"] == 5
    assert payload["latency_ms"] == 12.3


def test_log_with_fields_attaches_structured_fields(settings: Settings) -> None:
    patched = _with_logging(settings, level="DEBUG", console=True, file=False, json_format=True)
    setup_logging(patched)

    captured: dict = {}

    def _capture(record: logging.LogRecord) -> bool:
        captured.update(getattr(record, "fields", None) or {})
        return True

    logger = get_logger("auto_rag.test_fields")
    logger.addFilter(_capture)
    log_with_fields(logger, logging.INFO, "benchmark done", queries=100, p95_ms=41.2)
    assert captured == {"queries": 100, "p95_ms": 41.2}
