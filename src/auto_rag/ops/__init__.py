"""Operations tooling: health checks, config validation, benchmarking.

Production-readiness utilities for the AutoRAG deployment:

* :mod:`auto_rag.ops.checks`  - reusable runtime health checks.
* :mod:`auto_rag.ops.server`  - stdlib HTTP liveness/readiness probe server.
* :mod:`auto_rag.ops.stats`   - percentile/summary statistics.
* ``auto-rag-health``         - one-shot checks or the HTTP server (CLI).
* ``auto-rag-config-check``   - deep configuration validation (CLI).
* ``auto-rag-bench``          - retrieval / RAG latency benchmarks (CLI).
"""
