"""Tests for the eval-set loader and evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_rag.errors import ConfigurationError
from auto_rag.eval.loader import EvalExample, load_eval_set
from auto_rag.eval.runner import is_relevant, run_evaluation
from auto_rag.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, text="x", metadata={"source": source}, source=source
    )


def test_is_relevant_by_source_and_id() -> None:
    example = EvalExample(
        query="q", relevant_sources=("manual.pdf",), relevant_chunk_ids=("abc",)
    )
    assert is_relevant(_chunk("1", r"data\documents\manual.pdf"), example)
    assert is_relevant(_chunk("abc", "other.pdf"), example)
    assert not is_relevant(_chunk("2", "other.pdf"), example)


def test_run_evaluation_perfect() -> None:
    class _FakeRetriever:
        def retrieve(self, query, top_k=None) -> list[RetrievedChunk]:
            return [_chunk(f"r{i}", "manual.pdf") for i in range(3)]

    report = run_evaluation(
        _FakeRetriever(),
        [EvalExample(query="q", relevant_sources=("manual.pdf",))],
        top_k=3,
        pool_k=10,
    )
    assert report.metrics == {
        "hit@k": 1.0,
        "precision@k": 1.0,
        "recall@k": 1.0,
        "mrr@k": 1.0,
        "ndcg@k": 1.0,
    }
    query = report.queries[0]
    assert query.retrieved_ids == ["r0", "r1", "r2"]
    assert query.elapsed_ms >= 0


def test_run_evaluation_mixed_relevance() -> None:
    class _FakeRetriever:
        def retrieve(self, query, top_k=None) -> list[RetrievedChunk]:
            return [
                _chunk("a", "manual.pdf"),
                _chunk("b", "other.pdf"),
                _chunk("c", "manual.pdf"),
            ]

    report = run_evaluation(
        _FakeRetriever(),
        [EvalExample(query="q", relevant_sources=("manual.pdf",))],
        top_k=2,
        pool_k=10,
    )
    assert report.metrics["precision@k"] == 0.5
    assert report.metrics["recall@k"] == pytest.approx(0.5)
    assert report.metrics["mrr@k"] == 1.0


def test_run_evaluation_empty_relevant_pool() -> None:
    class _FakeRetriever:
        def retrieve(self, query, top_k=None) -> list[RetrievedChunk]:
            return [_chunk(f"r{i}", "other.pdf") for i in range(top_k)]

    report = run_evaluation(
        _FakeRetriever(),
        [EvalExample(query="q", relevant_sources=("manual.pdf",))],
        top_k=3,
        pool_k=10,
    )
    assert report.metrics["recall@k"] == 0.0
    assert report.metrics["hit@k"] == 0.0


def test_run_evaluation_report_to_dict() -> None:
    class _FakeRetriever:
        def retrieve(self, query, top_k=None) -> list[RetrievedChunk]:
            return [_chunk("a", "manual.pdf")]

    report = run_evaluation(
        _FakeRetriever(),
        [EvalExample(query="q", relevant_sources=("manual.pdf",))],
        top_k=1,
    )
    payload = report.to_dict()
    assert payload["top_k"] == 1
    assert "metrics" in payload
    assert payload["queries"][0]["query"] == "q"


def _write_eval_set(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_eval_set_valid(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    _write_eval_set(
        path,
        {
            "queries": [
                {
                    "query": "torque spec",
                    "relevant_sources": ["manual.pdf"],
                },
                {
                    "query": "misfire",
                    "relevant_chunk_ids": ["abc"],
                },
            ]
        },
    )
    examples = load_eval_set(path)
    assert len(examples) == 2
    assert examples[0].relevant_sources == ("manual.pdf",)
    assert examples[1].relevant_chunk_ids == ("abc",)


def test_load_eval_set_accepts_bare_list(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    _write_eval_set(path, [{"query": "q", "relevant_sources": ["manual.pdf"]}])
    assert len(load_eval_set(path)) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"queries": []},
        {"queries": [{"relevant_sources": ["manual.pdf"]}]},
        {"queries": [{"query": "q"}]},
        {"queries": ["not-an-object"]},
        "not-a-list-or-object",
    ],
)
def test_load_eval_set_rejects_malformed(tmp_path: Path, payload) -> None:
    path = tmp_path / "eval.json"
    _write_eval_set(path, payload)
    with pytest.raises(ConfigurationError):
        load_eval_set(path)


def test_load_eval_set_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_eval_set(tmp_path / "nope.json")


def test_eval_cli_main(tmp_path: Path, monkeypatch, capsys) -> None:
    import auto_rag.eval.cli as eval_cli

    eval_path = tmp_path / "eval.json"
    _write_eval_set(
        eval_path,
        {
            "queries": [
                {"query": "torque", "relevant_sources": ["manual.pdf"]},
                {"query": "misfire", "relevant_sources": ["dtc.pdf"]},
            ]
        },
    )

    class _FakeRetriever:
        def retrieve(self, query, top_k=None) -> list[RetrievedChunk]:
            return [_chunk(f"r{i}", "manual.pdf") for i in range(top_k)]

    monkeypatch.setattr(
        eval_cli, "get_settings", lambda: type("S", (), {"prepare_directories": lambda self: None})()
    )
    monkeypatch.setattr(eval_cli, "setup_logging", lambda settings: None)
    monkeypatch.setattr(eval_cli, "build_retriever", lambda settings: _FakeRetriever())

    assert eval_cli.main(["--eval-set", str(eval_path), "--top-k", "2"]) == 0
    captured = capsys.readouterr().out
    assert "hit@k" in captured
    assert "precision@k" in captured
