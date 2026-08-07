"""Runtime health checks for the AutoRAG stack.

Each check is a small, self-contained function returning a
:class:`CheckResult`. Both the CLI (:mod:`auto_rag.ops.health_cli`) and the
HTTP health server (:mod:`auto_rag.ops.server`) aggregate these checks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from auto_rag.config import Settings

__all__ = [
    "CheckResult",
    "check_database",
    "check_directories",
    "check_llm",
    "check_settings",
    "check_vector_store",
    "overall_ok",
    "run_checks",
]

_LIVENESS_CHECKS: frozenset[str] = frozenset({"settings", "directories"})
_READINESS_CHECKS: frozenset[str] = frozenset(
    {"settings", "directories", "database", "vector_store", "llm"}
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single health check."""

    name: str
    ok: bool
    detail: str
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
        }


def _measure(name: str, fn: Callable[[], tuple[bool, str]]) -> CheckResult:
    started = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - surface every failure as a result
        return CheckResult(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    return CheckResult(
        name=name,
        ok=bool(ok),
        detail=detail,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def check_settings(settings: Settings) -> CheckResult:
    """Report the resolved configuration summary."""
    return CheckResult(
        name="settings",
        ok=True,
        detail=(
            f"environment={settings.app.environment} version={settings.app.version} "
            f"db={settings.sqlite_path} chroma={settings.chroma_persist_dir}"
        ),
    )


def check_directories(settings: Settings) -> CheckResult:
    """Verify every runtime directory exists."""

    def _run() -> tuple[bool, str]:
        expected = (
            ("data", settings.paths.data_dir),
            ("documents", settings.paths.documents_dir),
            ("db", settings.paths.db_dir),
            ("logs", settings.paths.logs_dir),
        )
        missing = [label for label, path in expected if not path.is_dir()]
        if missing:
            return False, f"missing directories: {', '.join(missing)}"
        return True, "all runtime directories present"

    return _measure("directories", _run)


def check_database(settings: Settings) -> CheckResult:
    """Open, migrate and probe the SQLite database."""

    def _run() -> tuple[bool, str]:
        from auto_rag.db.connection import Database

        database = Database.from_settings(settings)
        try:
            database.initialize()
            count = database.scalar("SELECT COUNT(*) FROM conversations") or 0
        finally:
            database.close()
        return True, f"ok; conversations={count}"

    return _measure("database", _run)


def check_vector_store(settings: Settings, *, deep: bool = False) -> CheckResult:
    """Verify the Chroma collection is readable.

    ``deep=False`` opens the persisted store directly (no model loads).
    ``deep=True`` builds the full vector store stack including the embedding
    model, which is slow and only used on demand.
    """

    def _run() -> tuple[bool, str]:
        persist_dir = settings.chroma_persist_dir
        if not persist_dir.is_dir():
            return False, f"persist dir missing: {persist_dir}"
        if deep:
            from auto_rag.ingestion.cli_config import build_vector_store

            store = build_vector_store(settings)
            count = len(store.get_all_chunks(limit=1_000_000))
        else:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_dir))
            collection = client.get_collection(
                name=settings.vectorstore.collection_name
            )
            count = collection.count()
        return (
            True,
            f"collection={settings.vectorstore.collection_name} chunks={count}",
        )

    return _measure("vector_store", _run)


def check_llm(settings: Settings) -> CheckResult:
    """Verify the configured LLM backend is reachable."""

    def _run() -> tuple[bool, str]:
        from auto_rag.llm.factory import build_llm

        # Health probes must fail fast; clamp the client timeout.
        clamped = settings.llm.model_copy(
            update={"timeout_seconds": min(settings.llm.timeout_seconds, 5.0)}
        )
        llm = build_llm(clamped)
        if not llm.ping():
            return False, f"unreachable at {settings.llm.base_url}"
        return (
            True,
            f"{settings.llm.provider}/{settings.llm.model} reachable at "
            f"{settings.llm.base_url}",
        )

    return _measure("llm", _run)


def run_checks(
    settings: Settings,
    *,
    deep: bool = False,
    include_llm: bool = True,
) -> list[CheckResult]:
    """Run every configured health check in a stable order."""
    results = [
        check_settings(settings),
        check_directories(settings),
        check_database(settings),
        check_vector_store(settings, deep=deep),
    ]
    if include_llm:
        results.append(check_llm(settings))
    return results


def overall_ok(
    results: list[CheckResult],
    *,
    required: frozenset[str] | None = None,
) -> bool:
    """Return True when every ``required`` check is present and passing."""
    required = _READINESS_CHECKS if required is None else required
    by_name = {result.name: result for result in results}
    return all(
        (result := by_name.get(name)) is not None and result.ok for name in required
    )
