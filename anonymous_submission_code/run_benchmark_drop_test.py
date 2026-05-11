# -*- coding: utf-8 -*-
"""CLI entrypoint for benchmark capability drop tests."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-sided drop t-test for benchmark capability deltas.")
    parser.add_argument(
        "--input",
        default="benchmark_deltas_by_dataset.md",
        help="Benchmark delta table to read.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/benchmark_drop_ttest",
        help="Directory for output CSV/Markdown files.",
    )
    parser.add_argument(
        "--metric-column",
        default="Delta Acc",
        help="Delta column to test.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Significance threshold for drop decisions.",
    )
    parser.add_argument(
        "--include-overall-p-in-markdown",
        action="store_true",
        help="Add an Overall p-drop column to the Markdown table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Benchmark drop test] Args: {args}")
    from statistical_analysis.benchmark_drop_ttest import run

    outputs = run(args)
    for name, path in outputs.items():
        print(f"[Benchmark drop test] Wrote {name}: {path}")


if __name__ == "__main__":
    main()
