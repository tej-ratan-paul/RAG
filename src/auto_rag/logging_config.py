"""Structured logging configuration.

Initialises the root logger once with a console handler and/or a rotating
file handler, using either a human-readable or JSON formatter. All modules
should obtain loggers via :func:`get_logger`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from typing import Any, Final

from auto_rag.config import Settings
from auto_rag.constants import APP_NAME, APP_VERSION
from auto_rag.utils.paths import ensure_directory

__all__ = ["get_logger", "log_with_fields", "setup_logging"]

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%dT%H:%M:%S%z"

_configured: bool = False


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def __init__(self, *args: Any, service: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt or _DEFAULT_DATEFMT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.service:
            payload["service"] = self.service
        for extra in ("filename", "lineno", "funcName"):
            value = getattr(record, extra, None)
            if value:
                payload[extra] = value
        for key, value in (getattr(record, "fields", None) or {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger of the configured root logger."""
    return logging.getLogger(name)


def log_with_fields(
    logger: logging.Logger, level: int, message: str, **fields: Any
) -> None:
    """Log ``message`` including arbitrary ``fields`` in structured records."""
    logger.log(level, message, extra={"fields": fields})


def _root_level(settings: Settings) -> int:
    return getattr(logging, settings.logging.level.upper())


def setup_logging(settings: Settings) -> None:
    """Configure the root logger (idempotent; safe to call repeatedly).

    Args:
        settings: Application settings controlling handlers and format.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    level = _root_level(settings)
    root.setLevel(level)

    formatter: logging.Formatter
    if settings.logging.json_format:
        formatter = JsonFormatter(
            datefmt=_DEFAULT_DATEFMT, service=f"{APP_NAME} v{APP_VERSION}"
        )
    else:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    handlers: list[logging.Handler] = []
    if settings.logging.console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        handlers.append(console)

    if settings.logging.file:
        ensure_directory(settings.log_file_path.parent)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file_path,
            maxBytes=settings.logging.file_max_bytes,
            backupCount=settings.logging.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for handler in handlers:
        root.addHandler(handler)

    root.propagate = False
    _configured = True

    get_logger(__name__).info(
        "Logging initialised: level=%s console=%s file=%s json=%s",
        settings.logging.level,
        settings.logging.console,
        settings.logging.file,
        settings.logging.json_format,
    )
