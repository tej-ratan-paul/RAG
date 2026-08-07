"""Retrieval pipeline: dense + BM25 + MMR + rerank."""

from auto_rag.retrieval.bm25 import BM25Index, build_bm25_index, tokenize
from auto_rag.retrieval.filters import matches_filter, to_where
from auto_rag.retrieval.models import RetrievalFilter, RetrievedChunk
from auto_rag.retrieval.reranker import (
    CrossEncoderReranker,
    NoopReranker,
    Reranker,
    build_reranker,
)
from auto_rag.retrieval.retriever import (
    Retriever,
    mmr_select,
    reciprocal_rank_fusion,
)

__all__ = [
    "BM25Index",
    "build_bm25_index",
    "tokenize",
    "matches_filter",
    "to_where",
    "RetrievalFilter",
    "RetrievedChunk",
    "CrossEncoderReranker",
    "NoopReranker",
    "Reranker",
    "build_reranker",
    "Retriever",
    "mmr_select",
    "reciprocal_rank_fusion",
]
