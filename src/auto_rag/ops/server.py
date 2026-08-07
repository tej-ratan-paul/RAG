"""Minimal stdlib HTTP health server.

Serves liveness and readiness probes over JSON so orchestrators (Docker,
Kubernetes, load balancers) can monitor the deployment without Streamlit
running. Zero third-party dependencies; uses :mod:`http.server`.

Endpoints:

* ``GET /``       - service metadata and endpoint list.
* ``GET /health`` - liveness: process up and settings/directories present.
* ``GET /ready``  - readiness: full stack (DB, vector store, LLM).
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from auto_rag.config import Settings
from auto_rag.ops.checks import CheckResult, run_checks

logger = logging.getLogger(__name__)

__all__ = ["HealthServer", "make_handler"]


def _status(results: list[CheckResult], required: frozenset[str]) -> dict[str, Any]:
    checks = [result.to_dict() for result in results]
    healthy = all(result.ok for result in results if result.name in required)
    return {
        "status": "healthy" if healthy else "unhealthy",
        "healthy": healthy,
        "checks": checks,
    }


def make_handler(
    settings: Settings,
    *,
    deep: bool = False,
    include_llm: bool = True,
    service_name: str = "auto-rag",
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to ``settings``.

    ``deep`` enables the slow model-loading vector store check on ``/ready``.
    ``include_llm=False`` drops the LLM reachability check (useful for pure
    retrieval deployments).
    """
    liveness_required = frozenset({"settings", "directories"})
    readiness_required = frozenset(
        {"settings", "directories", "database", "vector_store"}
    )
    if include_llm:
        readiness_required = readiness_required | frozenset({"llm"})

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = self.path.split("?")[0]
            if path in ("/health", "/health/"):
                results = run_checks(settings, deep=False, include_llm=False)
                payload = _status(results, liveness_required)
            elif path in ("/ready", "/ready/"):
                results = run_checks(settings, deep=deep, include_llm=include_llm)
                payload = _status(results, readiness_required)
            elif path in ("/", ""):
                payload = {"service": service_name, "endpoints": ["/health", "/ready"]}
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("health http %s", fmt % args)

    return HealthHandler


class HealthServer:
    """A :class:`ThreadingHTTPServer` wrapper with graceful shutdown."""

    def __init__(
        self,
        settings: Settings,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        deep: bool = False,
        include_llm: bool = True,
        service_name: str = "auto-rag",
    ) -> None:
        self._settings = settings
        self._httpd = ThreadingHTTPServer(
            (host, port),
            make_handler(
                settings,
                deep=deep,
                include_llm=include_llm,
                service_name=service_name,
            ),
        )

    @property
    def host(self) -> str:
        """The bound host address."""
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        """The bound port (useful with ``port=0`` for ephemeral binds)."""
        return int(self._httpd.server_address[1])

    def serve_forever(self) -> None:
        """Serve until :meth:`shutdown` is called (blocking)."""
        logger.info("Health server listening on %s:%s", self.host, self.port)
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop serving and close the socket."""
        self._httpd.shutdown()
        self._httpd.server_close()
