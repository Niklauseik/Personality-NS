# -*- coding: utf-8 -*-
"""Stage-2 new-layout entrypoint (CLI).

Convenience wrapper for processing one or more new-layout model roots (contain `base/`).
Internally forwards to Stage-2 pipeline in `stage2/process_results.py`.
"""

from __future__ import annotations

import argparse
from typing import Sequence
from types import SimpleNamespace

from stage2.process_results import run as run_stage2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 (new layout): process one or more model roots (contain base/).")
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
        help='Glob pattern(s) (relative to CWD) to discover model roots, e.g. "*_newlayout". Can be repeated.',
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining model roots if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list which model roots would be processed.",
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
    model_roots = args.model_root or []
    print(f"[Stage-2/newlayout] Args: {args}")
    forwarded = SimpleNamespace(
        results_root=model_roots,
        results_glob=args.model_glob,
        continue_on_error=bool(args.continue_on_error),
        dry_run=bool(args.dry_run),
        skip_mbti=bool(args.skip_mbti),
        mbti_dataset=args.mbti_dataset,
        mbti_trials=args.mbti_trials,
        mbti_decode_method=args.mbti_decode_method,
        mbti_force=bool(args.mbti_force),
        mbti_include_invalid_pairs=bool(args.mbti_include_invalid_pairs),
        mbti_torch_dtype=args.mbti_torch_dtype,
        mbti_device_map=args.mbti_device_map,
        mbti_trust_remote_code=bool(args.mbti_trust_remote_code),
    )
    run_stage2(forwarded)


if __name__ == "__main__":
    main()
