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
    parser.add_argument(
        "--skip-mbti",
        action="store_true",
        help="Skip MBTI personality validation at the start of Stage-2.",
    )
    parser.add_argument(
        "--mbti-dataset",
        default="MBTI/data/MBTI_doubled_93.json",
        help="Path to the doubled MBTI JSON used for Stage-2 personality validation.",
    )
    parser.add_argument(
        "--mbti-trials",
        type=int,
        default=1,
        help="Number of MBTI validation trials per model. Logit decoding is deterministic, so 1 is usually enough.",
    )
    parser.add_argument(
        "--mbti-decode-method",
        choices=["logit", "generate"],
        default="logit",
        help="MBTI answer decoding method: next-token A/B logit comparison or greedy generation.",
    )
    parser.add_argument(
        "--mbti-force",
        action="store_true",
        help="Rerun MBTI validation even if result files already exist.",
    )
    parser.add_argument(
        "--mbti-include-invalid-pairs",
        action="store_true",
        help="Keep cross-dimension MBTI JSON items instead of skipping them.",
    )
    parser.add_argument(
        "--mbti-torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used for MBTI model loading.",
    )
    parser.add_argument(
        "--mbti-device-map",
        default="auto",
        help="Transformers device_map argument for MBTI model loading.",
    )
    parser.add_argument(
        "--mbti-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers for MBTI model loading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Stage-2] Args: {args}")
    run(args)


if __name__ == "__main__":
    main()
