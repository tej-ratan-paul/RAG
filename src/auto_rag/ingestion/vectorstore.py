"""Vector store (ChromaDB).

Manages the persistent collection of document chunks and their embeddings,
with metadata filtering. Query-time hybrid/MMR/rerank orchestration lives in
the retrieval package (Phase 4).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from auto_rag.errors import VectorStoreError
from auto_rag.ingestion.embeddings import EmbeddingProvider
from auto_rag.utils.paths import ensure_directory

logger = logging.getLogger(__name__)


def chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic, stable id for a chunk."""
    digest = hashlib.sha1(f"{source}:{chunk_index}".encode()).hexdigest()[:20]
    return f"{digest}:{chunk_index}"


class VectorStore:
    """ChromaDB-backed collection of embedded chunks."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        provider: EmbeddingProvider,
        distance: str = "cosine",
    ) -> None:
        import chromadb

        self.provider = provider
        self.distance = distance
        self.persist_dir = Path(persist_dir)
        ensure_directory(self.persist_dir)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance},
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not initialise collection: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def add_chunks(
        self,
        chunks: list,
        source: str,
    ) -> int:
        """Embed and store chunks. ``chunks`` are objects with ``.text`` and
        ``.metadata`` attributes (see :class:`auto_rag.ingestion.chunking.Chunk`).

        Returns the number of chunks indexed.
        """
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        vectors = self.provider.embed(texts)
        ids = [chunk_id(source, int(chunk.metadata["chunk_index"])) for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "source": source,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=vectors.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks: {exc}") from exc
        logger.info("Indexed %d chunks from %s", len(chunks), source)
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """Remove all chunks belonging to a source path."""
        try:
            self._collection.delete(where={"source": source})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete source {source}: {exc}") from exc

    def reset(self) -> None:
        """Delete all chunks in the collection."""
        try:
            ids = self._collection.get(include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset collection: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return self._collection.count()

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks with metadata and score.

        Each hit: ``{id, text, metadata, score}`` where ``score`` is a
        similarity in ``[0, 1]`` (higher is more relevant).
        """
        query_vector = self.provider.embed_query(query_text)
        try:
            result = self._collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        return _build_hits(result)

    def query_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a precomputed query vector (used by Phase 4)."""
        try:
            result = self._collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc
        return _build_hits(result)

    def get_all_chunks(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return stored chunks for inspection/evaluation."""
        data = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        hits: list[dict[str, Any]] = []
        for doc_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"], strict=False
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "score": None,
                }
            )
        return hits

    def get_embeddings(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Return stored embedding vectors for the given chunk ids."""
        if not ids:
            return {}
        try:
            data = self._collection.get(ids=ids, include=["embeddings"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to fetch embeddings: {exc}") from exc
        return {
            chunk_id: np.asarray(vector, dtype=np.float32)
            for chunk_id, vector in zip(
                data["ids"], data["embeddings"], strict=False
            )
        }


def _build_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a raw Chroma query result to uniform hit dicts."""
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for doc_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        hits.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "score": _to_similarity(distance),
            }
        )
    return hits


def _to_similarity(distance: float) -> float:
    """Convert Chroma distance to a similarity score in [0, 1]."""
    if distance >= 1.0:
        return 0.0
    if distance <= 0.0:
        return 1.0
    return round(1.0 - distance, 4)
