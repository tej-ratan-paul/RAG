# Deployment Guide

This guide covers running the AutoRAG Repair Assistant in a production-style
environment. Local (Windows / bare metal) usage stays as documented in the
main `README`.

## Architecture

```
                       ┌─────────────────────────────┐
   user ──► :8501 ──►  │ Streamlit UI (auto-rag-ui)   │
                       │      entrypoint.sh           │
                       │        │                     │
                       │        ├── auto-rag-health   │──► :8080 (/health, /ready)
                       │        └── auto_rag.rag      │
                       │             │                │
                       │             │                │
                       │    SQLite + Chroma (volume)  │
                       └─────────────┬───────────────┘
                                     │ LLM__BASE_URL
                              ┌──────▼──────┐
                              │ Ollama (:11434) │
                              └─────────────┘
```

- **SQLite** (`data/db/auto_rag.db`): conversations, vehicles, parts, DTCs,
  maintenance schedules, service history.
- **Chroma** (`data/db/chroma`): embedded document chunks.
- **Ollama**: LLM inference. Runs on the host, in the compose stack
  (`--profile with-ollama`), or anywhere reachable via `LLM__BASE_URL`.
- Embedding / reranker models download on first use into
  `data/.cache/huggingface` (set by `HF_HOME` inside the container).

## Container stack (recommended)

Requires **Docker Compose v2.24+**.

```bash
docker compose up --build
```

- Streamlit UI: <http://localhost:8501>
- Health liveness: <http://localhost:8080/health>
- Readiness: <http://localhost:8080/ready>

With a bundled Ollama backend:

```bash
docker compose --profile with-ollama up --build
```

Then point the app at it by adding `LLM__BASE_URL=http://ollama:11434` to your
`.env`. To use Ollama installed on the host instead, add `extra_hosts` +
`LLM__BASE_URL=http://host.docker.internal:11434` to the `auto-rag` service in
`docker-compose.yml`.

### Data persistence

The `auto-rag-data` named volume mounts `/app/data`. Documents ingested into
the container live on that volume; place source documents with
`docker compose cp <file> auto-rag:/app/data/documents/` or mount a local
directory:

```yaml
volumes:
  - auto-rag-data:/app/data
  - ./data/documents:/app/data/documents:ro
```

### First-run notes

1. Ingest documents: `docker compose exec auto-rag auto-rag-ingest ...`
   (or via the UI).
2. The container's healthcheck hits `/health` every 30s; it starts green once
   settings + directories are valid (model downloads may make the first
   request slow — `start_period` covers this).
3. Set `APP__ENVIRONMENT=production`, `LOGGING__JSON_FORMAT=true`, and
   `LOGGING__CONSOLE=false` in production so a log shipper tails the file.

## Bare-metal deployment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Validate config and readiness before serving traffic:
auto-rag-config-check --no-llm --format json
auto-rag-health --no-llm --format json

# Long-running processes:
auto-rag-health --serve --host 0.0.0.0 --port 8080 &   # probe server
auto-rag-ui                                            # Streamlit UI
```

Use a process manager (systemd, NSSM on Windows) to supervise both processes
and `restart: unless-stopped` semantics as in compose.

## Configuration

Settings come from environment variables / `.env` (see `.env.example`). Key
production knobs:

| Variable                    | Example value            | Purpose                                  |
| --------------------------- | ------------------------ | ---------------------------------------- |
| `APP__ENVIRONMENT`          | `production`             | Flags non-development deployments.       |
| `LLM__BASE_URL`             | `http://ollama:11434`    | LLM endpoint (compose network or host).  |
| `LLM__MODEL`                | `llama3.1`               | Pulled/available model on the Ollama side.|
| `LOGGING__JSON_FORMAT`      | `true`                   | Structured logs for shipping/alerting.   |
| `LOGGING__CONSOLE`          | `false`                  | Disable console when a file shipper runs.|
| `EMBEDDINGS__CACHE_ENABLED` | `true`                   | Speed up re-ingestion of unchanged text. |
| `HEALTH__HOST` / `HEALTH__PORT` | `0.0.0.0` / `8080`   | Probe server bind address.               |

## Health checks

`auto-rag-health` runs two kinds of probe:

- `/health` — **liveness**: settings load and runtime directories exist. No LLM
  dependency; used by the Docker `HEALTHCHECK`.
- `/ready` — **readiness**: adds the database and vector store; with `--deep`
  also verifies an actual vector query, and by default pings the LLM
  (`--no-llm` to skip).

One-shot mode (exit code 0 = healthy):

```bash
auto-rag-health --format json
```

## CI/CD

- `.github/workflows/ci.yml` — on every push/PR: ruff lint + format check,
  pytest on Python 3.11–3.13, and CLI smoke tests.
- `.github/workflows/deploy.yml` — on `v*` tags (or manual dispatch): builds the
  container image and pushes it to `ghcr.io/<repo>` with semver + SHA tags.

## Performance & evaluation

```bash
# Latency benchmarks (retrieval only, and full RAG):
auto-rag-bench --queries data/benchmarks/queries.txt --with-llm

# Retrieval quality against a labeled set:
auto-rag-eval --eval-set data/eval/my_set.json --top-k 5 --json-out report.json
```

A sample eval set ships with the package
(`auto_rag/eval/data/sample_eval_set.json`). Metrics: hit@k, precision@k,
recall@k, MRR@k, nDCG@k. Use `auto-rag-eval --help` for options.

## Security notes

- The container runs as a non-root `app` user; never switch to root in images
  layered on this one.
- `.env` is excluded from the build context (`.dockerignore`) and never
  committed.
- Keep Ollama on a private network; it exposes an unauthenticated API.
- Rotation is configured for file logs (`LOGGING__FILE_MAX_BYTES` /
  `LOGGING__FILE_BACKUP_COUNT`).
