# API Reference

The `auto_rag` package exposes a layered Python API plus eight console scripts.
All public classes live in `src/auto_rag`. Import them directly from their
module, e.g. `from auto_rag.rag.service import RAGService`.

## Console scripts

| Script                | Entrypoint                       | Purpose                                          |
| --------------------- | -------------------------------- | ------------------------------------------------ |
| `auto-rag-ingest`     | `auto_rag.ingestion.cli:main`    | Clean, chunk, embed, and store documents.        |
| `auto-rag-retrieve`   | `auto_rag.retrieval.cli:main`    | Query the vector store and show retrieved chunks.|
| `auto-rag-ask`        | `auto_rag.rag.cli:main`          | Ask a repair question; get a grounded, cited answer. |
| `auto-rag-ui`         | `auto_rag.ui.app:run`            | Launch the Streamlit web UI.                     |
| `auto-rag-health`     | `auto_rag.ops.health_cli:main`   | One-shot checks or a live HTTP health server.    |
| `auto-rag-config-check` | `auto_rag.ops.validate:main`   | Validate settings and runtime prerequisites.     |
| `auto-rag-bench`       | `auto_rag.ops.benchmark:main`   | Benchmark retrieval / RAG latency.               |
| `auto-rag-eval`        | `auto_rag.eval.cli:main`        | Evaluate retrieval quality against a labeled set.|

Run any script with `--help` for options.

## Configuration

Settings load from environment variables / `.env` using pydantic-settings with
nested `<SECTION>__<FIELD>` naming. The root model is `auto_rag.config.Settings`.
See `.env.example` for the full catalog. Highlights:

- **App**: `APP__NAME`, `APP__VERSION`, `APP__ENVIRONMENT`, `APP__DEBUG`.
- **Paths**: `PATHS__PROJECT_ROOT`, `PATHS__DATA_DIR`, `PATHS__DB_DIR`, ...
- **Database**: `DATABASE__PATH`, `DATABASE__FILENAME`.
- **Vector store**: `VECTORSTORE__BACKEND`, `VECTORSTORE__PERSIST_DIR`,
  `VECTORSTORE__COLLECTION_NAME`, `VECTORSTORE__DISTANCE`.
- **Embeddings**: `EMBEDDINGS__MODEL`, `EMBEDDINGS__DIMENSION`,
  `EMBEDDINGS__DEVICE`, `EMBEDDINGS__CACHE_ENABLED`, `EMBEDDINGS__CACHE_FILE`.
- **LLM**: `LLM__PROVIDER`, `LLM__BASE_URL`, `LLM__MODEL`,
  `LLM__TEMPERATURE`, `LLM__MAX_TOKENS`, `LLM__TIMEOUT_SECONDS`, `LLM__TOP_P`.
- **Retrieval**: `RETRIEVAL__TOP_K`, `RETRIEVAL__HYBRID_SEARCH`,
  `RETRIEVAL__HYBRID_TOP_K`, `RETRIEVAL__MMR`, `RETRIEVAL__MMR_LAMBDA_MULT`,
  `RETRIEVAL__MMR_FETCH_K`, `RETRIEVAL__RERANK`, `RETRIEVAL__RERANKER_MODEL`,
  `RETRIEVAL__RERANK_TOP_K`.
- **Chunking**: `CHUNKING__SIZE`, `CHUNKING__OVERLAP`.
- **Logging**: `LOGGING__LEVEL`, `LOGGING__JSON_FORMAT`, `LOGGING__CONSOLE`,
  `LOGGING__FILE`, `LOGGING__FILE_MAX_BYTES`, `LOGGING__FILE_BACKUP_COUNT`.
- **Health**: `HEALTH__HOST`, `HEALTH__PORT`.

## Core modules

### `auto_rag.config`

- `Settings` — root pydantic-settings model. Also exposes `embedding_cache_path`
  (`data/cache/embedding_cache.json`) and `prepare_directories()`.
- `HealthConfig`, `LoggingConfig`, `RetrievalConfig`, `LLMConfig`, ... — nested
  section models.

### `auto_rag.rag.service`

- `RAGService(retriever, llm, memory, *, prompt=None, default_top_k=None)` — the
  public facade over the agentic graph.
  - `ask(question, *, conversation_id=None, retrieval_filter=None, top_k=None,
    title=None) -> RAGResult`
  - `ask_stream(...)` — incremental answer text; final result on `last_result`.

### `auto_rag.rag.models`

- `RAGResult` — question, answer, citations, confidence, safety notes, timing,
  trace fields.
- `RetrievalFilter`, `QueryEvaluation`, `EvaluationReport` (in `auto_rag.eval`).

### `auto_rag.retrieval.retriever`

- `Retriever(vector_store, bm25_index=None, ...)` — hybrid dense + lexical
  retrieval with fusion, MMR, and cross-encoder reranking.
  - `retrieve(query, *, filter=None, top_k=5) -> list[Chunk]`
  - `last_timings: dict[str, float]` — per-stage millisecond timings
    (`dense`, `lexical`, `fusion`, `mmr`, `rerank`).

### `auto_rag.ingestion`

- `pipeline.IngestionPipeline` — clean → chunk → embed → store.
- `embeddings.get_embedding_provider(config, *, cache_path=None)` — builds the
  embedding provider, optionally wrapped in a persistent JSON cache.
- `embeddings.CachedEmbeddingProvider` — SHA-256 keyed cache with `cache_hits` /
  `cache_misses` counters and `flush()`.
- `vectorstore.VectorStore` / `build_vector_store` (in `cli_config.py`).
- `chunking.Chunk(text=..., metadata=...)`.

### `auto_rag.llm`

- `factory.build_llm(config)` — returns an `LLM` implementation from
  `LLM__PROVIDER` (`ollama` or an OpenAI-compatible endpoint).

### `auto_rag.ops`

- `checks.run_checks(settings, *, deep=False, include_llm=True) ->
  list[CheckResult]`; `overall_ok(results, required=...)`.
- `server.HealthServer(settings, host, port, *, deep, include_llm)` — stdlib
  HTTP server with `/`, `/health` (liveness), `/ready` (readiness).
- `validate.run_validation(settings)` — structural + runtime validation used by
  `auto-rag-config-check`.
- `benchmark` — retrieval and RAG benchmarking with per-stage timings.
- `stats.percentile`, `stats.summarize` — latency percentiles (p50/p95/p99).

### `auto_rag.eval`

- `loader.load_eval_set(path) -> list[EvalExample]` — labeled queries with
  `relevant_sources` / `relevant_chunk_ids`.
- `runner.run_evaluation(retriever, examples, *, top_k=5, pool_k=None) ->
  EvaluationReport` — aggregates hit@k, precision@k, recall@k, MRR@k, nDCG@k.
- `metrics.*` — the individual ranking metrics.
- `cli.main` — `auto-rag-eval`; ships `data/sample_eval_set.json` (8 labeled
  queries for the demo corpus).

### `auto_rag.logging_config`

- `setup_logging(settings)` — console + rotating file handlers.
- `JsonFormatter(service=...)` and `log_with_fields(logger, level, message,
  **fields)` — structured JSON logging.

## Programmatic example

```python
from auto_rag.config import get_settings
from auto_rag.ingestion.cli_config import build_vector_store
from auto_rag.ingestion.embeddings import get_embedding_provider
from auto_rag.retrieval.bm25 import BM25Index
from auto_rag.retrieval.reranker import build_reranker
from auto_rag.retrieval.retriever import Retriever
from auto_rag.rag.service import RAGService
from auto_rag.llm.factory import build_llm
from auto_rag.rag.memory import ConversationMemory

settings = get_settings()
settings.prepare_directories()
store = build_vector_store(settings)
provider = get_embedding_provider(
    settings.embeddings, cache_path=settings.embedding_cache_path
)
chunks = store.get_all_chunks(limit=100_000)
retriever = Retriever(
    vector_store=store,
    embedding_provider=provider,
    config=settings.retrieval,
    bm25_index=BM25Index(chunks),
    reranker=build_reranker(
        enabled=settings.retrieval.rerank,
        model_name=settings.retrieval.reranker_model,
        device=settings.embeddings.device,
    ),
)
service = RAGService(retriever, build_llm(settings.llm), ConversationMemory(settings))
result = service.ask("What does DTC P0300 indicate?")
print(result.answer)
```

## Stability note

This is pre-1.0 software; APIs may change between minor releases.
