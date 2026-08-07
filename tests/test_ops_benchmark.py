"""Tests for the benchmark CLI helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import auto_rag.ops.benchmark as benchmark
from auto_rag.ops.benchmark import DEFAULT_QUERIES, benchmark_rag, benchmark_retrieval, load_queries


def test_load_queries_falls_back() -> None:
    assert load_queries(None, ("a", "b")) == ["a", "b"]


def test_load_queries_from_file(tmp_path: Path) -> None:
    path = tmp_path / "queries.txt"
    path.write_text(
        "What is a P0300?\n\n# a comment\n  How to bleed brakes?\n", encoding="utf-8"
    )
    assert load_queries(str(path), DEFAULT_QUERIES) == [
        "What is a P0300?",
        "How to bleed brakes?",
    ]


def test_load_queries_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "queries.txt"
    path.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_queries(str(path), DEFAULT_QUERIES)


def test_benchmark_retrieval_aggregates(tmp_path: Path) -> None:
    class _FakeRetriever:
        def __init__(self) -> None:
            self.calls = 0
            self.last_timings = {"dense": 1.0, "rerank": 2.0}

        def retrieve(self, query, top_k=None) -> list:
            self.calls += 1
            return []

    retriever = _FakeRetriever()
    report = benchmark_retrieval(retriever, ["q1", "q2"], repeats=2)
    assert retriever.calls == 4
    assert set(report) == {"total_ms", "dense_ms", "rerank_ms"}
    assert report["total_ms"]["count"] == 4
    assert report["dense_ms"]["count"] == 4


def test_benchmark_rag_deletes_conversation(tmp_path: Path) -> None:
    class _FakeRepo:
        def __init__(self) -> None:
            self.deleted: list[int] = []

        def delete(self, conversation_id: int) -> bool:
            self.deleted.append(conversation_id)
            return True

    class _FakeService:
        def __init__(self) -> None:
            self.last_result = None

        def ask(self, query, conversation_id=None) -> None:
            self.last_result = SimpleNamespace(conversation_id=42)

    repo = _FakeRepo()
    report = benchmark_rag(_FakeService(), repo, ["q"], repeats=1)
    assert report["count"] == 1
    assert repo.deleted == [42]


def test_benchmark_rag_with_no_result(tmp_path: Path) -> None:
    class _FakeRepo:
        def delete(self, conversation_id: int) -> bool:
            return True

    class _FakeService:
        last_result = None

        def ask(self, query, conversation_id=None) -> None:
            pass

    report = benchmark_rag(_FakeService(), _FakeRepo(), ["q"], repeats=1)
    assert report["count"] == 1


def test_benchmark_main_requires_positive_repeats(tmp_path: Path, monkeypatch) -> None:
    settings = SimpleNamespace()
    settings.prepare_directories = lambda: None
    monkeypatch.setattr(benchmark, "get_settings", lambda: settings)
    monkeypatch.setattr(benchmark, "setup_logging", lambda settings: None)
    with pytest.raises(SystemExit):
        benchmark.main(["--repeats", "0"])
