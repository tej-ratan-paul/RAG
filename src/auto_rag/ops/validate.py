"""``auto-rag-config-check``: validate the deployment configuration.

Goes beyond :mod:`auto_rag.ops.checks`: every configuration section is
inspected (supported providers, directory writability, backend choices) and
then the runtime reachability checks (database, vector store, LLM) run.

Usage::

    auto-rag-config-check                 # text report (exit 0/1)
    auto-rag-config-check --format json   # machine-readable report
    auto-rag-config-check --no-llm        # skip LLM reachability
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

from auto_rag.config import Settings, get_settings
from auto_rag.logging_config import get_logger, setup_logging
from auto_rag.ops.checks import run_checks

logger = get_logger(__name__)

_SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("ollama", "openai", "openai_compat")
_SUPPORTED_VECTOR_BACKENDS: tuple[str, ...] = ("chroma", "qdrant")
_SUPPORTED_EMBEDDING_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class ValidationItem:
    """Outcome of one configuration validation step."""

    section: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "ok": self.ok, "detail": self.detail}


def _writable(label: str, path: Any) -> ValidationItem:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".auto_rag_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return ValidationItem(label, True, str(path))
    except OSError as exc:
        return ValidationItem(label, False, f"{path}: {exc}")


def validate_sections(settings: Settings) -> list[ValidationItem]:
    """Run structural (no network) validations over every section."""
    items: list[ValidationItem] = [
        ValidationItem(
            "app",
            True,
            f"name={settings.app.name} version={settings.app.version} "
            f"environment={settings.app.environment}",
        ),
        ValidationItem("paths", True, f"project_root={settings.paths.project_root}"),
        _writable("paths.data", settings.paths.data_dir),
        _writable("paths.documents", settings.paths.documents_dir),
        _writable("paths.db", settings.paths.db_dir),
        _writable("paths.logs", settings.paths.logs_dir),
        ValidationItem(
            "database",
            True,
            f"path={settings.sqlite_path} journal={settings.database.journal_mode}",
        ),
        ValidationItem(
            "vectorstore.backend",
            settings.vectorstore.backend in _SUPPORTED_VECTOR_BACKENDS,
            f"backend={settings.vectorstore.backend} collection="
            f"{settings.vectorstore.collection_name} distance="
            f"{settings.vectorstore.distance}",
        ),
        ValidationItem(
            "embeddings.device",
            settings.embeddings.device in _SUPPORTED_EMBEDDING_DEVICES,
            f"model={settings.embeddings.model} dimension="
            f"{settings.embeddings.dimension} device={settings.embeddings.device} "
            f"cache_enabled={settings.embeddings.cache_enabled}",
        ),
        ValidationItem(
            "llm.provider",
            settings.llm.provider in _SUPPORTED_LLM_PROVIDERS,
            f"provider={settings.llm.provider} model={settings.llm.model} "
            f"base_url={settings.llm.base_url}",
        ),
        ValidationItem(
            "retrieval",
            True,
            f"top_k={settings.retrieval.top_k} hybrid={settings.retrieval.hybrid_search} "
            f"mmr={settings.retrieval.mmr} rerank={settings.retrieval.rerank}",
        ),
        ValidationItem(
            "chunking",
            True,
            f"size={settings.chunking.size} overlap={settings.chunking.overlap}",
        ),
        ValidationItem(
            "logging",
            True,
            f"level={settings.logging.level} json={settings.logging.json_format}",
        ),
        ValidationItem(
            "health",
            True,
            f"host={settings.health.host} port={settings.health.port}",
        ),
    ]
    return items


def run_validation(settings: Settings, *, include_llm: bool) -> list[ValidationItem]:
    """Structural validation followed by runtime reachability checks."""
    items = validate_sections(settings)
    for result in run_checks(settings, include_llm=include_llm):
        items.append(
            ValidationItem(result.name, result.ok, result.detail)
        )
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-config-check",
        description="Validate the AutoRAG deployment configuration.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Report format.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM reachability validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    items = run_validation(settings, include_llm=not args.no_llm)
    if args.format == "json":
        print(json.dumps([item.to_dict() for item in items], indent=2))
    else:
        for item in items:
            marker = "OK  " if item.ok else "FAIL"
            print(f"[{marker}] {item.section:<22} {item.detail}")
        failures = sum(1 for item in items if not item.ok)
        print(f"\n{failures} validation failure(s).")
    return 1 if any(not item.ok for item in items) else 0


if __name__ == "__main__":
    sys.exit(main())
