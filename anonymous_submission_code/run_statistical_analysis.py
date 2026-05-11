# -*- coding: utf-8 -*-
"""CLI entrypoint for significance summaries and plots."""

from __future__ import annotations

import argparse
from typing import Sequence

DEFAULT_MIN_P = 1e-300
DEFAULT_P_THRESHOLD = 0.05


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize significance results and draw plots for new-layout roots.")
    parser.add_argument(
        "--model-root",
        nargs="+",
        default=None,
        help="One or more new-layout model roots that contain base/.",
    )
    parser.add_argument(
        "--model-glob",
        action="append",
        default=[],
        help='Glob pattern(s) to discover new-layout model roots, such as "*_newlayout".',
    )
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    parser.add_argument("--p-threshold", type=float, default=DEFAULT_P_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Statistical analysis] Args: {args}")
    from statistical_analysis.significance import run

    run(args)


if __name__ == "__main__":
    main()
