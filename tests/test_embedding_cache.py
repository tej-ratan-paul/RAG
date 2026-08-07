"""Tests for the disk-backed embedding cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from auto_rag.ingestion.embeddings import (
    CachedEmbeddingProvider,
    DeterministicHashEmbeddingProvider,
    get_embedding_provider,
)
from auto_rag.ingestion.vectorstore import VectorStore


@pytest.fixture
def cached(tmp_path) -> CachedEmbeddingProvider:
    inner = DeterministicHashEmbeddingProvider(dimension=64)
    return CachedEmbeddingProvider(inner, cache_path=tmp_path / "cache.json")


def test_missing_vectors_computed_and_cached(cached: CachedEmbeddingProvider) -> None:
    vectors = cached.embed(["alpha", "beta", "alpha"])
    assert vectors.shape == (3, 64)
    assert cached.cache_misses == 3
    assert cached.cache_hits == 0
    assert np.allclose(vectors[0], vectors[2])


def test_subsequent_embed_hits_only(cached: CachedEmbeddingProvider) -> None:
    first = cached.embed(["alpha", "beta"])
    second = cached.embed(["alpha", "beta"])
    assert np.array_equal(first, second)
    assert cached.cache_misses == 2
    assert cached.cache_hits == 2


def test_embed_query_uses_cache(cached: CachedEmbeddingProvider) -> None:
    query = "brake caliper torque"
    vec1 = cached.embed_query(query)
    vec2 = cached.embed_query(query)
    assert vec1.shape == (64,)
    assert np.array_equal(vec1, vec2)
    assert cached.cache_misses == 1
    assert cached.cache_hits == 1


def test_empty_list(cached: CachedEmbeddingProvider) -> None:
    assert cached.embed([]).shape == (0, 64)


def test_persistence_round_trip(tmp_path) -> None:
    cache_file = tmp_path / "cache.json"
    first = CachedEmbeddingProvider(
        DeterministicHashEmbeddingProvider(dimension=64), cache_path=cache_file
    )
    vectors = first.embed(["alpha", "beta"])
    first.flush()
    assert cache_file.is_file()

    second = CachedEmbeddingProvider(
        DeterministicHashEmbeddingProvider(dimension=64), cache_path=cache_file
    )
    assert second.embed(["alpha", "beta"]).shape == (2, 64)
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert np.array_equal(vectors, second.embed(["alpha", "beta"]))


def test_cache_file_format(tmp_path) -> None:
    cache_file = tmp_path / "cache.json"
    provider = CachedEmbeddingProvider(
        DeterministicHashEmbeddingProvider(dimension=64), cache_path=cache_file
    )
    provider.embed(["alpha"])
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["dimension"] == 64
    assert len(payload["vectors"]) == 1


def test_dimension_mismatch_ignores_existing_cache(tmp_path) -> None:
    cache_file = tmp_path / "cache.json"
    first = CachedEmbeddingProvider(
        DeterministicHashEmbeddingProvider(dimension=32), cache_path=cache_file
    )
    first.embed(["alpha"])
    second = CachedEmbeddingProvider(
        DeterministicHashEmbeddingProvider(dimension=64), cache_path=cache_file
    )
    assert second.cache_misses == 0
    assert second.embed(["alpha"]).shape == (1, 64)
    assert second.cache_misses == 1


def test_build_vector_store_passes_cache_path(
    settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_get(config, *, cache_path):
        captured["cache_path"] = cache_path
        return DeterministicHashEmbeddingProvider(dimension=64)

    import auto_rag.ingestion.cli_config as cli_config

    monkeypatch.setattr(cli_config, "get_embedding_provider", fake_get)
    store = VectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="docs",
        provider=DeterministicHashEmbeddingProvider(dimension=64),
    )
    monkeypatch.setattr(cli_config, "VectorStore", lambda **_: store)
    assert cli_config.build_vector_store(settings) is store
    assert captured["cache_path"] == settings.embedding_cache_path


def test_get_embedding_provider_wraps_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_rag.config import EmbeddingsConfig

    fake_inner = DeterministicHashEmbeddingProvider(dimension=64)
    monkeypatch.setattr(
        "auto_rag.ingestion.embeddings._provider_from", lambda **_: fake_inner
    )

    enabled = EmbeddingsConfig(cache_enabled=True)
    wrapped = get_embedding_provider(enabled, cache_path="cache.json")
    assert isinstance(wrapped, CachedEmbeddingProvider)

    disabled = EmbeddingsConfig(cache_enabled=False)
    assert get_embedding_provider(disabled, cache_path="cache.json") is fake_inner
