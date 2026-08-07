"""Retrieval orchestration.

Combines the dense and lexical channels, fuses their rankings with Reciprocal
Rank Fusion, applies Maximal Marginal Relevance for diversity, and optionally
reranks with a cross-encoder. Exposes :func:`reciprocal_rank_fusion` and
:func:`mmr_select` as pure, unit-testable helpers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from auto_rag.config import RetrievalConfig
from auto_rag.ingestion.embeddings import EmbeddingProvider, l2_normalise
from auto_rag.ingestion.vectorstore import VectorStore
from auto_rag.retrieval.filters import to_where
from auto_rag.retrieval.models import RetrievalFilter, RetrievedChunk
from auto_rag.retrieval.reranker import NoopReranker, Reranker

logger = logging.getLogger(__name__)

_RRF_K: int = 60  # smoothing constant for reciprocal rank fusion


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int = _RRF_K,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one via Reciprocal Rank Fusion.

    Each item's score is ``sum(1 / (k + rank))`` across the lists it appears
    in; items unique to one channel still rank well. Duplicate ids are merged.
    """
    fused: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    result = [by_id[chunk_id].with_score(score) for chunk_id, score in ordered]
    return result[:top_k] if top_k is not None else result


def mmr_select(
    candidates: list[RetrievedChunk],
    query_vector: np.ndarray,
    embeddings: dict[str, np.ndarray],
    lambda_mult: float,
    top_k: int,
) -> list[RetrievedChunk]:
    """Greedy Maximal Marginal Relevance selection over candidates.

    Balances relevance to the query (``lambda_mult``) against redundancy with
    already-selected chunks (``1 - lambda_mult``). Requires stored embedding
    vectors for at least one candidate; candidates without vectors are dropped.
    """
    pairs = [(chunk, embeddings[chunk.id]) for chunk in candidates if chunk.id in embeddings]
    if not pairs:
        return []
    selected_chunks = [chunk for chunk, _ in pairs]
    matrix = l2_normalise(np.stack([vector for _, vector in pairs]).astype(np.float32))
    query_vec = l2_normalise(np.asarray(query_vector, dtype=np.float32))

    query_similarity = matrix @ query_vec
    chosen: list[int] = []
    result: list[RetrievedChunk] = []

    while len(result) < top_k and len(chosen) < len(selected_chunks):
        best_index: int | None = None
        best_score = -np.inf
        for index in range(len(selected_chunks)):
            if index in chosen:
                continue
            relevance = float(query_similarity[index])
            if chosen:
                redundancy = max(
                    float(matrix[index] @ matrix[j]) for j in chosen
                )
                score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            else:
                score = relevance
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
        result.append(selected_chunks[best_index].with_score(float(best_score)))

    return result


class Retriever:
    """Orchestrates dense + lexical retrieval with MMR and reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
        bm25_index: Any = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.provider = embedding_provider
        self.config = config or RetrievalConfig()
        self.bm25_index = bm25_index
        self.reranker = reranker or NoopReranker()
        self.last_timings: dict[str, float] = {}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Run the full retrieval pipeline and return ranked chunks.

        Args:
            query: The user's natural-language question.
            top_k: Final result count (falls back to ``config.top_k``).
            retrieval_filter: Optional metadata narrowing for both channels.

        Returns:
            Ranked chunks, most relevant first.
        """
        config = self.config
        final_k = top_k or config.top_k
        pool_k = config.hybrid_top_k if config.hybrid_search else final_k

        started = time.perf_counter()
        where = to_where(retrieval_filter) if retrieval_filter else None
        query_vector = self.provider.embed_query(query)
        dense_hits = self.vector_store.query_vectors(
            query_vector, pool_k, where=where
        )
        dense_chunks = [RetrievedChunk.from_hit(hit) for hit in dense_hits]
        timings = {"dense": (time.perf_counter() - started) * 1000}

        if config.hybrid_search and self.bm25_index is not None:
            started = time.perf_counter()
            lexical = self.bm25_index.search(
                query, top_k=pool_k, retrieval_filter=retrieval_filter
            )
            timings["lexical"] = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            candidates = reciprocal_rank_fusion(
                dense_chunks, lexical, top_k=config.mmr_fetch_k
            )
            timings["fusion"] = (time.perf_counter() - started) * 1000
            logger.debug(
                "Hybrid retrieval: %d dense, %d lexical, %d fused",
                len(dense_chunks),
                len(lexical),
                len(candidates),
            )
        else:
            candidates = dense_chunks[: config.mmr_fetch_k]

        # MMR diversity pass over the fused candidate pool.
        started = time.perf_counter()
        if config.mmr and len(candidates) > final_k:
            embeddings = self.vector_store.get_embeddings(
                [chunk.id for chunk in candidates]
            )
            candidates = mmr_select(
                candidates,
                query_vector,
                embeddings,
                config.mmr_lambda_mult,
                final_k,
            )
        else:
            candidates = candidates[:final_k]
        timings["mmr"] = (time.perf_counter() - started) * 1000

        # Optional cross-encoder rerank.
        started = time.perf_counter()
        if config.rerank:
            candidates = self.reranker.rerank(query, candidates, config.rerank_top_k)
        timings["rerank"] = (time.perf_counter() - started) * 1000

        self.last_timings = timings
        return candidates[:final_k]
