# -*- coding: utf-8 -*-
"""Stage-1 entrypoint (CLI).

Handles CLI args/logging, then calls implementation in `stage1/train_and_test.py`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from stage1.train_and_test import run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1: Train models and generate raw benchmark/sentiment results.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dimension",
        choices=["energy", "information", "decision", "execution"],
        help="Train both subtypes for the specified MBTI dimension.",
    )
    mode_group.add_argument(
        "--pair",
        nargs=2,
        metavar=("TYPE_A", "TYPE_B"),
        help="Train a pair of custom personality codes (e.g., ST NF or ENTP ISFJ).",
    )
    parser.add_argument("--model-path", required=True, help="Base model checkpoint path.")
    parser.add_argument("--output-root", default="dpo_outputs", help="Directory to store trained checkpoints.")
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run benchmark evaluations (default: enabled). Use --no-benchmark to skip.",
    )
    parser.add_argument(
        "--base-sentiment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run sentiment inference for the base model (default: enabled). Use --no-base-sentiment to skip.",
    )
    parser.add_argument(
        "--sentiment-runs",
        type=int,
        default=1,
        help="How many times to run sentiment inference (fully repeated runs). Default: 1.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Directory where evaluation results are stored (default: auto like results-N-S or results-ST-NF).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Stage-1] Args: {args}")
    run(args)


if __name__ == "__main__":
    main()
