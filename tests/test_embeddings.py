"""Tests for embedding providers."""

from __future__ import annotations

import numpy as np

from auto_rag.ingestion.embeddings import (
    DeterministicHashEmbeddingProvider,
    l2_normalise,
    resolve_device,
)


def test_deterministic_provider_dimensions() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    vectors = provider.embed(["alpha", "beta", "alpha"])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(vectors[0], vectors[2])  # deterministic


def test_deterministic_provider_returns_unit_vectors() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=64)
    vectors = provider.embed(["some text", "other text"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_query_vector_shape() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=32)
    query = provider.embed_query("camshaft position sensor")
    assert query.shape == (32,)


def test_embed_empty_list() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=32)
    assert provider.embed([]).shape == (0, 32)

def test_l2_normalise_safe_with_zero_vectors() -> None:
    matrix = np.zeros((2, 4), dtype=np.float32)
    result = l2_normalise(matrix)
    assert np.isfinite(result).all()
    assert np.allclose(result, 0.0)


def test_resolve_device_auto() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}
    assert resolve_device("cpu") == "cpu"
