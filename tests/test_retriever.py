"""Tests for retrieval orchestration: fusion, MMR, and the Retriever."""

from __future__ import annotations

import numpy as np
import pytest

from auto_rag.config import RetrievalConfig
from auto_rag.ingestion.embeddings import l2_normalise
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.models import RetrievalFilter, RetrievedChunk
from auto_rag.retrieval.reranker import NoopReranker
from auto_rag.retrieval.retriever import Retriever, mmr_select, reciprocal_rank_fusion


def _chunk(chunk_id: str, text: str, score: float = 0.0, **meta) -> RetrievedChunk:
    metadata = {"source": f"{chunk_id}.pdf", **meta}
    return RetrievedChunk(id=chunk_id, text=text, metadata=metadata, score=score)


def test_rrf_merges_and_deduplicates() -> None:
    dense = [_chunk("a", "engine"), _chunk("b", "brakes")]
    lexical = [_chunk("b", "brakes"), _chunk("c", "oil")]
    fused = reciprocal_rank_fusion(dense, lexical)
    ids = [chunk.id for chunk in fused]
    assert ids == ["b", "a", "c"]
    assert len(fused) == 3


def test_rrf_respects_top_k() -> None:
    dense = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
    fused = reciprocal_rank_fusion(dense, top_k=2)
    assert [chunk.id for chunk in fused] == ["a", "b"]


def test_rrf_assigns_fused_score() -> None:
    dense = [_chunk("a", "x", score=0.9)]
    fused = reciprocal_rank_fusion(dense)
    assert fused[0].score == pytest.approx(1.0 / 61)


def test_mmr_picks_diverse_set() -> None:
    query = l2_normalise(np.array([1.0, 0.0], dtype=np.float32))
    vectors = {
        "a": l2_normalise(np.array([1.0, 0.001], dtype=np.float32)),
        "b": l2_normalise(np.array([1.0, 0.002], dtype=np.float32)),
        "c": l2_normalise(np.array([0.8, 0.6], dtype=np.float32)),
    }
    candidates = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
    # Low lambda heavily penalises the near-duplicate "b", favouring "c".
    selected = mmr_select(candidates, query, vectors, lambda_mult=0.3, top_k=2)
    ids = [chunk.id for chunk in selected]
    assert ids == ["a", "c"]


def test_mmr_missing_embeddings_dropped() -> None:
    query = l2_normalise(np.array([1.0, 0.0], dtype=np.float32))
    vectors = {"a": l2_normalise(np.array([0.99, 0.01], dtype=np.float32))}
    candidates = [_chunk("a", "x"), _chunk("missing", "y")]
    selected = mmr_select(candidates, query, vectors, lambda_mult=0.7, top_k=2)
    assert [chunk.id for chunk in selected] == ["a"]


# ------------------------------------------------------------------ #
# Retriever integration (deterministic embeddings, offline)
# ------------------------------------------------------------------ #
def _retriever(vector_store, *, config=None, **bm25_chunks) -> Retriever:
    bm25 = BM25Index(list(bm25_chunks.values())) if bm25_chunks else None
    return Retriever(
        vector_store=vector_store,
        embedding_provider=vector_store.provider,
        config=config or RetrievalConfig(rerank=False),
        bm25_index=bm25,
        reranker=NoopReranker(),
    )


def test_retriever_dense_only(vector_store) -> None:
    vector_store.add_chunks(
        [
            _chunk_obj("brake pad replacement procedure", 0),
            _chunk_obj("engine oil change steps", 1),
        ],
        source="manual.pdf",
    )
    retriever = _retriever(
        vector_store,
        config=RetrievalConfig(hybrid_search=False, mmr=False, rerank=False),
    )
    hits = retriever.retrieve("brake pad replacement", top_k=1)
    assert len(hits) == 1
    assert "brake" in hits[0].text.lower()
    assert hits[0].source == "manual.pdf"


def test_retriever_hybrid_lexical_boost(vector_store) -> None:
    vector_store.add_chunks(
        [
            _chunk_obj("replace the camshaft position sensor P0340", 0),
            _chunk_obj("inspect the ignition system wiring", 1),
        ],
        source="manual.pdf",
    )
    # Build the lexical index from the store's own chunks so ids align with dense.
    bm25 = BM25Index(vector_store.get_all_chunks())
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=vector_store.provider,
        config=RetrievalConfig(hybrid_search=True, mmr=False, rerank=False),
        bm25_index=bm25,
        reranker=NoopReranker(),
    )
    hits = retriever.retrieve("P0340 sensor", top_k=2)
    assert hits[0].metadata["source"] == "manual.pdf"
    assert "P0340" in hits[0].text


def test_retriever_metadata_filter(vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("brake pad replacement", 0, make="Toyota")],
        source="toyota.pdf",
    )
    vector_store.add_chunks(
        [_chunk_obj("brake pad replacement", 0, make="Honda")],
        source="honda.pdf",
    )
    retriever = _retriever(
        vector_store,
        config=RetrievalConfig(hybrid_search=False, mmr=False, rerank=False),
    )
    hits = retriever.retrieve(
        "brake pads",
        top_k=5,
        retrieval_filter=RetrievalFilter(make="Toyota"),
    )
    assert len(hits) == 1
    assert hits[0].metadata["make"] == "Toyota"


def test_retriever_mmr_engaged(vector_store) -> None:
    vector_store.add_chunks(
        [
            _chunk_obj("how to change the oil filter", 0),
            _chunk_obj("change the engine oil filter", 1),
            _chunk_obj("replace the air filter element", 2),
        ],
        source="manual.pdf",
    )
    retriever = _retriever(
        vector_store,
        config=RetrievalConfig(
            top_k=2, hybrid_search=False, mmr=True, mmr_fetch_k=3,
            rerank=False, rerank_top_k=2,
        ),
    )
    hits = retriever.retrieve("oil filter change", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_retriever_scores_descending(vector_store) -> None:
    vector_store.add_chunks(
        [
            _chunk_obj("brake disc rotor replacement guide", 0),
            _chunk_obj("coolant flush procedure details", 1),
        ],
        source="manual.pdf",
    )
    retriever = _retriever(
        vector_store,
        config=RetrievalConfig(hybrid_search=False, mmr=False, rerank=False),
    )
    hits = retriever.retrieve("brake rotor replacement", top_k=2)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_retriever_records_step_timings(vector_store) -> None:
    vector_store.add_chunks(
        [_chunk_obj("brake pad replacement procedure", 0)],
        source="manual.pdf",
    )
    bm25 = BM25Index(vector_store.get_all_chunks())
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=vector_store.provider,
        config=RetrievalConfig(hybrid_search=True, mmr=True, rerank=True),
        bm25_index=bm25,
        reranker=NoopReranker(),
    )
    retriever.retrieve("brake pad replacement", top_k=1)
    assert set(retriever.last_timings) == {
        "dense",
        "lexical",
        "fusion",
        "mmr",
        "rerank",
    }
    for elapsed in retriever.last_timings.values():
        assert elapsed >= 0


def _chunk_obj(text: str, index: int, **meta) -> dict:
    from auto_rag.ingestion.chunking import Chunk

    metadata = {"chunk_index": index, "title": "x", "doc_type": "service_manual",
                "make": "", "model": "", "year": "", "engine": "", "vin": "", **meta}
    return Chunk(text=text, metadata=metadata)
