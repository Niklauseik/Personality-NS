# -*- coding: utf-8 -*-
"""Stage-3 entrypoint (CLI).

Handles CLI args/logging, then calls implementation in `stage3/significance.py`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from stage3.significance import DEFAULT_MIN_P, DEFAULT_P_THRESHOLD, run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-3: summarize significance + draw plots (new layout only).")
    parser.add_argument(
        "--model-root",
        nargs="+",
        default=None,
        help="One or more new-layout model roots (contain base/).",
    )
    parser.add_argument(
        "--model-glob",
        action="append",
        default=[],
        help='Glob pattern(s) to discover new-layout model roots, e.g. "*_newlayout".',
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=DEFAULT_MIN_P,
        help="Lower bound for p-value when computing -log10(p).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=DEFAULT_P_THRESHOLD,
        help="Significance threshold for summary conclusion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list which model roots would be processed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Stage-3] Args: {args}")
    run(args)


if __name__ == "__main__":
    main()
