"""``auto-rag-health``: run health checks once or serve HTTP probes.

Usage::

    auto-rag-health                          # one-shot report (exit 0/1)
    auto-rag-health --format json            # machine-readable report
    auto-rag-health --serve --port 8080      # blocking HTTP probe server
"""

from __future__ import annotations

import argparse
import json
import sys

from auto_rag.config import get_settings
from auto_rag.logging_config import get_logger, setup_logging
from auto_rag.ops.checks import CheckResult, overall_ok, run_checks
from auto_rag.ops.server import HealthServer

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-health",
        description="Check the AutoRAG deployment health.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the HTTP health server instead of one-shot checks.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address for --serve (default: HEALTH__HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for --serve (default: HEALTH__PORT).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Report format (one-shot mode).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Load models so the vector store check exercises the full stack.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM reachability check (retrieval-only deployment).",
    )
    return parser


def _render(results: list[CheckResult], fmt: str) -> str:
    if fmt == "json":
        return json.dumps([result.to_dict() for result in results], indent=2)
    lines: list[str] = []
    for result in results:
        marker = "OK  " if result.ok else "FAIL"
        lines.append(
            f"[{marker}] {result.name:<12} ({result.duration_ms:7.1f} ms) "
            f"{result.detail}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    include_llm = not args.no_llm

    if args.serve:
        server = HealthServer(
            settings,
            host=args.host or settings.health.host,
            port=args.port or settings.health.port,
            deep=args.deep,
            include_llm=include_llm,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - interactive path
            server.shutdown()
        return 0

    results = run_checks(settings, deep=args.deep, include_llm=include_llm)
    print(_render(results, args.format))
    required = frozenset(
        {"settings", "directories", "database", "vector_store", "llm"}
    )
    if not include_llm:
        required = required - frozenset({"llm"})
    if not overall_ok(results, required=required):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
