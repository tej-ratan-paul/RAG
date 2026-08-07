# syntax=docker/dockerfile:1
# AutoRAG Repair Assistant runtime image.
#
# Build:  docker build -t auto-rag:0.1.0 .
# Run:    docker compose up
#
# Notes:
# * Uses the CPU build of PyTorch to keep the image small.
# * Embedding / reranker models download on first use into the /app/data
#   volume (see HF_HOME). Pre-seed the volume for fully offline operation.
# * PATHS__PROJECT_ROOT is fixed to /app so runtime state stays on the volume.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATHS__PROJECT_ROOT=/app \
    HF_HOME=/app/data/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System prerequisites: build tools for wheels without prebuilt binaries,
# curl for diagnostics, and a non-root runtime user.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home /app app

WORKDIR /app

# Install the project (CPU-only torch keeps the image lean).
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install .

# Runtime entrypoint and probe helpers.
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
COPY scripts/healthcheck.py ./scripts/healthcheck.py
RUN chmod +x scripts/entrypoint.sh \
    && mkdir -p /app/data/documents /app/data/db /app/data/logs \
    && chown -R app:app /app

USER app

EXPOSE 8501 8080

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python scripts/healthcheck.py http://127.0.0.1:8080/health

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["auto-rag-ui"]
