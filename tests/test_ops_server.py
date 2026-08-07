"""Tests for the HTTP health probe server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from auto_rag.config import Settings
from auto_rag.ingestion.vectorstore import VectorStore
from auto_rag.ops.server import HealthServer


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


def _start_server(settings: Settings, **kwargs) -> tuple[HealthServer, threading.Thread]:
    server = HealthServer(settings, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_health_endpoints(tmp_path: Path, deterministic_embeddings) -> None:
    from auto_rag.ingestion.chunking import Chunk

    settings = _make_settings(tmp_path)
    settings.prepare_directories()
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.vectorstore.collection_name,
        provider=deterministic_embeddings,
        distance="cosine",
    )
    store.add_chunks(
        [Chunk(text="brake pad thickness", metadata={"chunk_index": 0})],
        source="manual.pdf",
    )

    server, thread = _start_server(settings, include_llm=False)
    base = f"http://127.0.0.1:{server.port}"
    try:
        meta = _get(f"{base}/")
        assert meta["endpoints"] == ["/health", "/ready"]

        health = _get(f"{base}/health")
        assert health["healthy"] is True
        assert health["status"] == "healthy"

        ready = _get(f"{base}/ready")
        assert ready["healthy"] is True
        check_names = {item["name"] for item in ready["checks"]}
        assert {"settings", "directories", "database", "vector_store"} <= check_names

        try:
            _get(f"{base}/missing")
            raise AssertionError("expected HTTP 404")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ready_reports_llm_failure(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings(tmp_path)
    settings.prepare_directories()

    import auto_rag.ops.server as server_module
    from auto_rag.ops.checks import CheckResult

    def fake_run_checks(settings, *, deep=False, include_llm=True):
        results = [
            CheckResult("settings", True, "ok"),
            CheckResult("directories", True, "ok"),
            CheckResult("database", True, "ok"),
            CheckResult("vector_store", True, "ok"),
            CheckResult("llm", False, "unreachable at http://localhost:11434"),
        ]
        return results

    monkeypatch.setattr(server_module, "run_checks", fake_run_checks)

    server, thread = _start_server(settings, include_llm=True)
    base = f"http://127.0.0.1:{server.port}"
    try:
        ready = _get(f"{base}/ready")
        assert ready["healthy"] is False
        assert ready["status"] == "unhealthy"

        health = _get(f"{base}/health")
        assert health["healthy"] is True  # liveness ignores the LLM
    finally:
        server.shutdown()
        thread.join(timeout=5)
