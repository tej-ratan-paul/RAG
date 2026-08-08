# AutoRAG Repair Assistant

An **agentic RAG (Retrieval-Augmented Generation)** assistant that answers
automobile repair questions using OEM service manuals, DTC trouble codes, TSBs,
wiring diagrams, and structured workshop data (parts, labor rates, service
history, maintenance schedules).

Every answer is **grounded in retrieved sources** with inline citations,
confidence scoring, and safety notes — it does not guess.

> Status: pre-1.0 (alpha). APIs may change between minor releases.

---

## Table of Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Web UI](#web-ui)
- [Development](#development)
- [Performance & evaluation](#performance--evaluation)
- [Health & monitoring](#health--monitoring)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [License](#license)

---

## Features

**Document ingestion**

- Loads PDF and plain-text documents (`pypdf`), with automatic detection of
  document type (service manual, repair manual, DTC, TSB, wiring diagram).
- Loads **CSV files** and **SQL sources** (SQLite files or remote
  Postgres/MySQL URLs) as row-per-chunk tabular data; credentials in stored
  source labels are redacted.
- Extracts vehicle metadata (make, model, year, engine, VIN) from text.
- Recursive chunking with token-aware sizes and configurable overlap.
- Deduplication by file hash (or content fingerprint for SQL snapshots);
  re-indexing only changed sources (`--force` to override).
- Embedding cache (JSON, keyed by text hash) to speed up re-ingestion.

**Retrieval**

- **Hybrid search**: dense embeddings (sentence-transformers `all-MiniLM-L6-v2`)
  fused with lexical BM25 via reciprocal-rank fusion.
- **MMR** diversity re-ranking and optional **cross-encoder reranking**
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Metadata filters (make, model, year, document type).
- Per-stage latency timings exposed for benchmarking.

**Agentic answer generation**

- LangGraph workflow: retrieve → ground → generate, with conversation history.
- Answers cite sources inline `[1]`, list confidence, and emit safety notes.
- Streaming output support.
- Durable conversation memory in SQLite (context across follow-ups).

**Structured workshop data**

- SQLite-backed tables: vehicles, parts, labor operations, DTC codes,
  maintenance schedules, and service history — seeded with realistic demo data.

**Operations & tooling**

- HTTP **health server** (liveness `/health`, readiness `/ready`) for probes.
- **Config validation**, **latency benchmarking**, and **retrieval
  evaluation** (hit@k, precision@k, recall@k, MRR@k, nDCG@k) CLI tools.
- Structured JSON logging, rolling log files, pydantic-settings configuration.
- Streamlit **web UI** and Docker containerization.

---

## How it works

```
                        ┌───────────────────────────────┐
   question ──────────► │ Retriever                     │
                        │   dense (MiniLM embeddings)   │
                        │   + BM25 lexical              │
                        │   → RRF fusion                │
                        │   → MMR diversity             │
                        │   → cross-encoder rerank      │
                        └──────────────┬────────────────┘
                                       │ top-k chunks (with sources)
                        ┌──────────────▼────────────────┐
                        │ Agentic graph (LangGraph)     │
                        │   history + retrieved context │
                        │   → LLM (Ollama / OpenAI-     │
                        │     compatible)               │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │ Answer + citations +          │
                        │ confidence + safety notes     │
                        └───────────────────────────────┘
```

Documents are chunked and embedded into a **vector store** (Chroma by default;
Qdrant backend supported). Structured data lives in **SQLite**. The LLM never
sees anything outside the retrieved context.

---

## Repository layout

```
src/auto_rag/
  config.py             pydantic-settings configuration
  ingestion/            cleaning, chunking, embeddings, vector store, pipeline
  retrieval/            BM25, dense retrieval, MMR, reranker, filters
  rag/                  agentic graph, prompts, citations, memory, service
  llm/                  Ollama + OpenAI-compatible LLM adapters
  db/                   SQLite schema, migrations, repositories, seeder
  ops/                  health server, config validation, benchmarking
  eval/                 retrieval evaluation metrics, CLI, sample eval set
  ui/                   Streamlit web UI
  logging_config.py     structured logging setup
tests/                  pytest suite (unit + end-to-end)
data/                   runtime data: documents/, db/, logs/, benchmarks/
docs/                   api.md, deployment.md
```

---

## Requirements

- **Python 3.11+** (tested on 3.14)
- **Ollama** running locally (or an OpenAI-compatible endpoint) for answer
  generation — see [LLM setup](#llm-setup)
- Internet on first run to download embedding / reranker models
  (cached locally afterwards)
- Optional: Docker + Docker Compose v2.24+ for containerized deployment

### LLM setup

Install [Ollama](https://ollama.com) and pull a model (default `llama3.1`):

```bash
ollama pull llama3.1
```

Answers are generated only when the LLM is reachable; retrieval, ingestion,
health checks, and the UI work without it.

---

## Installation

Create a virtual environment and install the package:

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the environment template and adjust:

```bash
cp .env.example .env
```

---

## Configuration

All settings load from environment variables / `.env` using nested
`<SECTION>__<FIELD>` naming. See `.env.example` for the full catalog. The most
commonly adjusted variables:

| Variable                  | Default                          | Purpose                                   |
| ------------------------- | -------------------------------- | ----------------------------------------- |
| `LLM__BASE_URL`           | `http://localhost:11434`         | Ollama / OpenAI-compatible endpoint.      |
| `LLM__MODEL`              | `llama3.1`                       | LLM model used for answers.               |
| `LLM__TEMPERATURE`        | `0.1`                            | Lower = more grounded, less creative.     |
| `RETRIEVAL__TOP_K`        | `5`                              | Chunks used per answer.                   |
| `RETRIEVAL__RERANK`       | `true`                           | Enable cross-encoder reranking.           |
| `EMBEDDINGS__CACHE_ENABLED`| `false`                          | Cache embeddings by text hash.            |
| `LOGGING__JSON_FORMAT`    | `false`                          | Structured JSON logs (set `true` in prod).|
| `HEALTH__HOST` / `HEALTH__PORT` | `0.0.0.0` / `8080`          | Health server bind address.               |

Runtime state (SQLite DB, Chroma, logs) lives under `data/`, which is git-ignored.

---

## Quick start

All commands are installed as console scripts by `pip install -e .`.

**1. Seed structured demo data and ingest the demo corpus**

```powershell
auto-rag-ingest --seed
auto-rag-ingest --directory data/documents
```

The repo ships three demo documents: a Toyota Camry service manual (PDF),
a Honda Civic maintenance schedule (TXT), and a DTC P0300 codesheet (PDF).

To also index tabular workshop data, point ingestion at a CSV file or a SQL
source (SQLite file or remote URL). Each row becomes its own retrievable chunk:

```powershell
auto-rag-ingest --csv data/parts.csv
auto-rag-ingest --sqlite data/demo/workshop.db --table parts
auto-rag-ingest --sql-url "postgresql://user:pass@host:5432/workshop" --table vehicles
```

**2. Ask a grounded question**

```powershell
auto-rag-ask --query "What does DTC P0300 indicate?"
```

```powershell
auto-rag-ask --query "When should brake pads be inspected?" --make Toyota --year 2018
```

**3. Launch the web UI**

```powershell
auto-rag-ui
```

Open <http://localhost:8501>.

---

## CLI reference

| Command                | Purpose                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `auto-rag-ingest`      | Ingest documents into the vector store (see below).           |
| `auto-rag-retrieve`    | Query the vector store and print top chunks with scores.      |
| `auto-rag-ask`         | Ask a grounded question; get a cited, streamable answer.      |
| `auto-rag-ui`          | Launch the Streamlit web UI.                                  |
| `auto-rag-health`      | One-shot health checks or a live HTTP probe server.           |
| `auto-rag-config-check`| Validate configuration and runtime prerequisites.             |
| `auto-rag-bench`       | Benchmark retrieval / RAG latency.                            |
| `auto-rag-eval`        | Evaluate retrieval quality against a labeled set.             |

### `auto-rag-ingest`

```
--directory DIR   ingest all documents in a directory
--file FILE       ingest a single document
--csv FILE        ingest a CSV file (one chunk per row)
--sqlite FILE     ingest rows from a SQLite database file
--sql-url URL     ingest rows from a remote SQL source (postgresql://, mysql://)
--table TABLE     table to read (SELECT * FROM <table>)
--query SQL       raw SQL query (takes precedence over --table)
--limit N         cap the number of SQL rows ingested
--doc-type TYPE   override document type (service_manual, repair_manual,
                  dtc, tsb, wiring_diagram, tabular)
--force           re-index sources already in the store
--reset           delete all indexed chunks first
--seed            seed demo structured data into SQLite
```

File sources (`--directory` / `--file` / `--csv`) and SQL sources
(`--sqlite` / `--sql-url`) are mutually exclusive; `--table`, `--query`, and
`--limit` apply only to SQL ingestion. Remote SQL requires the matching
optional driver (`psycopg2` for Postgres, `pymysql` for MySQL) — SQLite files
need none.

### `auto-rag-retrieve`

```
--query TEXT      natural-language question (required)
--top-k N         number of results
--make / --model / --year / --doc-type   metadata filters
--no-rerank       disable cross-encoder reranking
```

### `auto-rag-ask`

```
--query TEXT            repair question (required)
--make / --model / --year / --doc-type   metadata filters
--conversation-id N     continue an existing conversation (memory/context)
--top-k N               retrieval chunks used
--stream                stream tokens as they are generated
--provider / --model-name / --base-url   LLM overrides
--no-rerank             disable reranking
```

### `auto-rag-health`

```
--serve                run the HTTP probe server (blocking)
--host / --port        bind address (defaults: HEALTH__HOST / HEALTH__PORT)
--format text|json     report format (one-shot mode)
--deep                 exercise the full stack (loads models)
--no-llm               skip the LLM reachability check
```

One-shot mode exits `0` when healthy, `1` otherwise. In `--serve` mode it
exposes `GET /health` (liveness) and `GET /ready` (readiness).

### `auto-rag-config-check`

```
--format text|json     report format
--no-llm               skip LLM reachability validation
```

### `auto-rag-bench`

```
--queries FILE   one query per line (# comments and blanks ignored)
--repeats N      times to time each query (default 3)
--with-llm       also time end-to-end RAG answers (needs reachable LLM)
--json-out FILE  write the JSON report to a file
```

### `auto-rag-eval`

```
--eval-set FILE  labeled eval set JSON (default: packaged sample set)
--top-k N        evaluation cutoff (default 5)
--pool-k N       candidate pool for recall (default top_k * 5)
--json-out FILE  write the JSON report
--limit N        evaluate only the first N queries
```

---

## Web UI

The Streamlit UI (`auto-rag-ui`) provides:

- A chat interface with streaming answers, citations, and safety notes.
- Conversation history with the ability to start/continue threads.
- A configuration page to inspect and adjust retrieval settings at runtime.

---

## Development

```bash
# Lint (ruff: E, F, W, I, UP, B, SIM, ASYNC)
ruff check src tests

# Run the full test suite (unit + E2E)
python -m pytest
```

The suite currently contains **294 passing tests** covering ingestion,
retrieval, the agentic graph, LLM adapters, database, configuration, the ops
tools, and evaluation.

---

## Performance & evaluation

```bash
# Latency benchmarks (retrieval only)
auto-rag-bench --queries data/benchmarks/queries.txt

# End-to-end RAG latency (requires a reachable LLM)
auto-rag-bench --queries data/benchmarks/queries.txt --with-llm --json-out report.json

# Retrieval quality against a labeled eval set
auto-rag-eval --json-out eval_report.json
```

Example results on the demo corpus (local machine, MiniLM embeddings):

- Retrieval total **p50 ≈ 75 ms** (dense ~17 ms, rerank ~55 ms).
- Eval metrics at `top_k=5`: **precision@k 0.92, recall@k 1.00, MRR@k 1.00,
  nDCG@k 1.00** (sample set, 8 queries).

A sample labeled eval set ships with the package at
`src/auto_rag/eval/data/sample_eval_set.json`.

---

## Health & monitoring

```bash
# One-shot checks (exit code 0 = healthy)
auto-rag-health --no-llm
auto-rag-health --format json

# Live probe server (for orchestrators / k8s / docker healthchecks)
auto-rag-health --serve --host 0.0.0.0 --port 8080
```

Endpoints:

| Endpoint   | Kind      | Checks                                                     |
| ---------- | --------- | ---------------------------------------------------------- |
| `/health`  | liveness  | Settings load, runtime directories exist (LLM-independent).|
| `/ready`   | readiness | Settings, directories, database, vector store, LLM ping.   |

---

## Deployment

See **[docs/deployment.md](docs/deployment.md)** for the full deployment guide,
including:

- Docker / Docker Compose stack (`docker compose up --build`).
- Bundled Ollama profile (`--profile with-ollama`).
- GitHub Actions CI (`ci.yml`) and GHCR image publishing (`deploy.yml`).
- Production configuration tips (structured logs, embedding cache, health
  probes).

---

## Documentation

- [docs/api.md](docs/api.md) — Python API reference and programmatic usage.
- [docs/deployment.md](docs/deployment.md) — container and bare-metal deployment.
- `.env.example` — every configuration variable with inline comments.

---

## License

MIT. See `LICENSE` (if present) for details.

