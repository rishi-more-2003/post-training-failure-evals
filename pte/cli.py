"""Command-line interface: `pte-eval` (or `python -m pte.cli`)."""

from __future__ import annotations

import argparse
import sys

from pte.config import load_config
from pte.evaluators import all_evaluator_names
from pte.runner import run_eval
from pte.utils import load_env, setup_logging

logger = setup_logging()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pte-eval",
        description="Post-training failure-mode evaluation harness (Tinker-backed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the eval suite from a YAML config.")
    p_run.add_argument("--config", "-c", required=True, help="Path to eval YAML config.")
    p_run.add_argument(
        "--suites",
        "-s",
        nargs="*",
        default=None,
        help="Subset of suites to run (default: all in config / all registered).",
    )
    p_run.add_argument("--limit", "-n", type=int, default=None, help="Cap examples per suite.")
    p_run.add_argument("--output-dir", "-o", default=None, help="Override results directory.")

    sub.add_parser("list-suites", help="List available evaluator suites.")

    args = parser.parse_args(argv)

    if args.command == "list-suites":
        print("Available suites:")
        for name in all_evaluator_names():
            print(f"  - {name}")
        return 0

    if args.command == "run":
        load_env()
        cfg = load_config(args.config)
        if args.suites:
            cfg.suites = args.suites
        if args.limit is not None:
            cfg.limit = args.limit
        if args.output_dir:
            cfg.output_dir = args.output_dir
        result = run_eval(cfg)
        print(f"\nResults written to: {result['out_dir']}")
        print("Composite failure scores (lower = safer):")
        for m, v in sorted(result["summary"]["composite_failure"].items(), key=lambda kv: kv[1]):
            print(f"  {m:>12}: {v:.3f}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
