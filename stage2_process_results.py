# -*- coding: utf-8 -*-
"""Stage-2 entrypoint (CLI).

Handles CLI args/logging, then calls implementation in `stage2/process_results.py`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from stage2.process_results import run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2: Process results, evaluate, and draw charts.")
    parser.add_argument(
        "--results-root",
        nargs="+",
        default=None,
        help="One or more results directories (e.g., results-NS-first-run).",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        default=[],
        help='Glob pattern(s) (relative to CWD) to discover results directories, e.g. "results-*". Can be repeated.',
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining results roots if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list which results roots would be processed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Stage-2] Args: {args}")
    run(args)


if __name__ == "__main__":
    main()
