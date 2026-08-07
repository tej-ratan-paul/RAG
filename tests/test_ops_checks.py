"""Tests for runtime health checks."""

from __future__ import annotations

from pathlib import Path

from auto_rag.config import Settings
from auto_rag.ingestion.vectorstore import VectorStore
from auto_rag.ops.checks import (
    check_database,
    check_directories,
    check_llm,
    check_settings,
    check_vector_store,
    overall_ok,
    run_checks,
)


def make_settings(tmp_path: Path) -> Settings:
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


def test_check_settings_reports_environment(tmp_path: Path) -> None:
    result = check_settings(make_settings(tmp_path))
    assert result.ok
    assert "environment=" in result.detail
    assert result.to_dict()["name"] == "settings"


def test_check_directories_after_prepare(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.prepare_directories()
    result = check_directories(settings)
    assert result.ok


def test_check_directories_reports_missing(tmp_path: Path) -> None:
    result = check_directories(make_settings(tmp_path))
    assert not result.ok
    assert "missing" in result.detail


def test_check_database_initialises_and_counts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.prepare_directories()
    result = check_database(settings)
    assert result.ok
    assert "conversations=0" in result.detail
    assert (settings.sqlite_path).is_file()


def test_check_vector_store_shallow(tmp_path: Path, deterministic_embeddings) -> None:
    from auto_rag.ingestion.chunking import Chunk

    settings = make_settings(tmp_path)
    settings.prepare_directories()
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.vectorstore.collection_name,
        provider=deterministic_embeddings,
        distance="cosine",
    )
    store.add_chunks(
        [Chunk(text="brake pad thickness", metadata={"chunk_index": 0, "title": "x"})],
        source="manual.pdf",
    )
    result = check_vector_store(settings)
    assert result.ok
    assert "chunks=1" in result.detail


def test_check_vector_store_missing_dir(tmp_path: Path) -> None:
    result = check_vector_store(make_settings(tmp_path))
    assert not result.ok
    assert "missing" in result.detail


def test_check_vector_store_deep(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.prepare_directories()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    class _FakeStore:
        def get_all_chunks(self, limit: int) -> list[dict]:
            assert limit == 1_000_000
            return [{"id": "x"}]

    monkeypatch.setattr(
        "auto_rag.ingestion.cli_config.build_vector_store", lambda settings: _FakeStore()
    )
    result = check_vector_store(settings, deep=True)
    assert result.ok
    assert "chunks=1" in result.detail


def test_check_llm_ok(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)

    class _FakeLLM:
        def ping(self) -> bool:
            return True

    monkeypatch.setattr(
        "auto_rag.llm.factory.build_llm", lambda config: _FakeLLM()
    )
    result = check_llm(settings)
    assert result.ok
    assert "reachable" in result.detail


def test_check_llm_unreachable(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)

    class _FakeLLM:
        def ping(self) -> bool:
            return False

    monkeypatch.setattr(
        "auto_rag.llm.factory.build_llm", lambda config: _FakeLLM()
    )
    result = check_llm(settings)
    assert not result.ok
    assert "unreachable" in result.detail


def test_run_checks_and_overall_ok(tmp_path: Path, deterministic_embeddings) -> None:
    from auto_rag.ingestion.chunking import Chunk

    settings = make_settings(tmp_path)
    settings.prepare_directories()
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.vectorstore.collection_name,
        provider=deterministic_embeddings,
        distance="cosine",
    )
    store.add_chunks(
        [Chunk(text="brake pad", metadata={"chunk_index": 0})],
        source="manual.pdf",
    )
    results = run_checks(settings, include_llm=False)
    names = [result.name for result in results]
    assert names == ["settings", "directories", "database", "vector_store"]
    assert overall_ok(results, required=frozenset(names))


def test_overall_ok_requires_all_present() -> None:
    from auto_rag.ops.checks import CheckResult

    results = [CheckResult("settings", True, "ok"), CheckResult("directories", True, "ok")]
    assert overall_ok(results, required=frozenset({"settings", "directories"}))
    assert not overall_ok(results, required=frozenset({"settings", "database"}))
