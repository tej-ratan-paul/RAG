"""Tests for the Chroma vector store."""

from __future__ import annotations

from auto_rag.ingestion.chunking import Chunk
from auto_rag.ingestion.vectorstore import VectorStore, chunk_id


def _chunk(text: str, index: int, **meta) -> Chunk:
    return Chunk(text=text, metadata={"chunk_index": index, **meta})


def test_chunk_id_is_stable() -> None:
    assert chunk_id("doc.pdf", 3) == chunk_id("doc.pdf", 3)
    assert chunk_id("doc.pdf", 3) != chunk_id("doc.pdf", 4)


def test_add_and_count(vector_store: VectorStore) -> None:
    vector_store.add_chunks(
        [
            _chunk("camshaft position sensor location", 0),
            _chunk("torque spec 15 Nm", 1),
        ],
        source="manual.pdf",
    )
    assert vector_store.count() == 2


def test_similarity_search_returns_top_hit(vector_store: VectorStore) -> None:
    vector_store.add_chunks(
        [
            _chunk("oil filter change procedure", 0),
            _chunk("front brake pad replacement", 1),
        ],
        source="manual.pdf",
    )
    hits = vector_store.similarity_search("brake pad replacement", top_k=2)
    assert hits[0]["metadata"]["chunk_index"] == 1
    assert hits[0]["score"] > hits[1]["score"]


def test_metadata_filter(vector_store: VectorStore) -> None:
    vector_store.add_chunks([_chunk("alternator wiring", 0)], source="a.pdf")
    vector_store.add_chunks([_chunk("alternator wiring", 0)], source="b.pdf")
    hits = vector_store.similarity_search(
        "alternator", top_k=5, where={"source": "a.pdf"}
    )
    assert len(hits) == 1
    assert hits[0]["metadata"]["source"] == "a.pdf"


def test_delete_source(vector_store: VectorStore) -> None:
    vector_store.add_chunks([_chunk("engine removal", 0)], source="engine.pdf")
    vector_store.add_chunks([_chunk("transmission removal", 0)], source="trans.pdf")
    vector_store.delete_source("engine.pdf")
    assert vector_store.count() == 1


def test_reset_clears_everything(vector_store: VectorStore) -> None:
    vector_store.add_chunks([_chunk("something", 0)], source="x.pdf")
    vector_store.reset()
    assert vector_store.count() == 0


def test_upsert_deduplicates_by_id(vector_store: VectorStore) -> None:
    vector_store.add_chunks([_chunk("same content", 0)], source="doc.pdf")
    vector_store.add_chunks([_chunk("same content", 0)], source="doc.pdf")
    assert vector_store.count() == 1
