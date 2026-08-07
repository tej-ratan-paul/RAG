"""Embedding generation.

Wraps HuggingFace sentence-transformers behind a small :class:`EmbeddingProvider`
protocol so the rest of the application (and tests) can swap in alternative
implementations without changing call sites.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from auto_rag.config import EmbeddingsConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Produces dense vectors for text."""

    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float array of shape ``(len(texts), dimension)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single query vector of shape ``(dimension,)``."""
        ...


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to ``cuda`` or ``cpu``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero vectors)."""
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix if norm == 0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HuggingFaceEmbeddingProvider:
    """Embeddings via a sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        dimension: int = 0,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved = resolve_device(device)
        logger.info("Loading embedding model %s on %s", model_name, resolved)
        self.model = SentenceTransformer(model_name, device=resolved)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return l2_normalise(vectors) if self.normalize else vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class DeterministicHashEmbeddingProvider:
    """Deterministic pseudo-embeddings for tests and offline demos.

    Produces stable, well-separated vectors by hashing character n-grams.
    Not semantically meaningful; only for exercising the pipeline.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalise(np.array([self._vector(t) for t in texts], dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        grams = [text[i : i + 3] for i in range(len(text) - 2)] or [text]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            vector[int(digest[0]) % self.dimension] += int.from_bytes(digest[1:3], "big")
        return vector.tolist()


class CachedEmbeddingProvider:
    """Disk-backed embedding cache wrapping another provider.

    Embeddings are keyed by the SHA-256 digest of the source text. Cached
    vectors skip model inference entirely, which accelerates re-indexing of
    unchanged documents and repeated query embeddings.

    The cache file is JSON::

        {"version": 1, "dimension": 384, "vectors": {"<hex-hash>": [...]}}

    Writes are atomic (temp file + rename) and guarded by a lock.
    """

    _CACHE_VERSION: Final[int] = 1

    def __init__(self, provider: EmbeddingProvider, cache_path: str | Path) -> None:
        self._provider = provider
        self.dimension = provider.dimension
        self.cache_path = Path(cache_path)
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self.cache_hits = 0
        self.cache_misses = 0
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != self._CACHE_VERSION:
                return
            if payload.get("dimension") != self.dimension:
                return
            self._vectors = dict(payload.get("vectors") or {})
        except (OSError, ValueError):
            self._vectors = {}

    def flush(self) -> None:
        """Atomically persist any pending vectors to disk."""
        if not self._dirty:
            return
        payload = {
            "version": self._CACHE_VERSION,
            "dimension": self.dimension,
            "vectors": self._vectors,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.cache_path)
        self._dirty = False

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        digests = [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ]
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        missing_indices: list[int] = []
        missing_digests: list[str] = []
        missing_texts: list[str] = []
        with self._lock:
            for index, digest in enumerate(digests):
                cached = self._vectors.get(digest)
                if cached is not None:
                    output[index] = cached
                    self.cache_hits += 1
                else:
                    missing_indices.append(index)
                    missing_digests.append(digest)
                    missing_texts.append(texts[index])
        if missing_texts:
            vectors = self._provider.embed(missing_texts)
            with self._lock:
                for row, index, digest in zip(
                    vectors, missing_indices, missing_digests, strict=False
                ):
                    stored = row.tolist()
                    self._vectors[digest] = stored
                    output[index] = stored
                self.cache_misses += len(missing_texts)
                self._dirty = True
            self.flush()
        return output

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def _provider_from(
    model: str,
    device: str,
    batch_size: int,
    normalize: bool,
    dimension: int,
) -> EmbeddingProvider:
    """Cached provider construction from hashable arguments."""
    return HuggingFaceEmbeddingProvider(
        model_name=model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        dimension=dimension,
    )


def get_embedding_provider(
    config: EmbeddingsConfig, *, cache_path: str | Path | None = None
) -> EmbeddingProvider:
    """Return a cached embedding provider built from config.

    When ``cache_enabled`` is set in the config and a ``cache_path`` is
    provided, the underlying provider is wrapped in a persistent
    :class:`CachedEmbeddingProvider`.
    """
    provider = _provider_from(
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        normalize=config.normalize,
        dimension=config.dimension,
    )
    if config.cache_enabled and cache_path is not None:
        return CachedEmbeddingProvider(provider, cache_path=cache_path)
    return provider
