"""``auto-rag-eval``: score retrieval quality against a labeled eval set.

Usage::

    auto-rag-eval                                   # built-in demo eval set
    auto-rag-eval --eval-set eval/custom.json --top-k 5
    auto-rag-eval --json-out data/benchmarks/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from auto_rag.config import get_settings
from auto_rag.eval.loader import load_eval_set
from auto_rag.eval.metrics import metric_labels
from auto_rag.eval.runner import run_evaluation
from auto_rag.logging_config import get_logger, setup_logging
from auto_rag.ops.benchmark import build_retriever

logger = get_logger(__name__)

DEFAULT_EVAL_SET = Path(__file__).parent / "data" / "sample_eval_set.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-rag-eval",
        description="Evaluate AutoRAG retrieval quality against a labeled eval set.",
    )
    parser.add_argument(
        "--eval-set",
        default=str(DEFAULT_EVAL_SET),
        help="Path to the labeled eval set JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Evaluation cutoff k (default: 5).",
    )
    parser.add_argument(
        "--pool-k",
        type=int,
        default=None,
        help="Candidate pool size used for recall (default: top_k * 5).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write the full JSON report to this path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N queries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.prepare_directories()
    setup_logging(settings)

    examples = load_eval_set(args.eval_set)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        print("No queries to evaluate.", file=sys.stderr)
        return 1

    retriever = build_retriever(settings)
    report = run_evaluation(retriever, examples, top_k=args.top_k, pool_k=args.pool_k)

    print(
        f"Retrieval evaluation (top_k={report.top_k}, pool_k={report.pool_k}, "
        f"n={len(report.queries)})"
    )
    for label in metric_labels():
        print(f"  {label:<12} {report.metrics[label]:.4f}")

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Report written to %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
