# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.pipeline_utils import ordered_model_entries
from MBTI.code.mbti_eval import (
    MBTIRunConfig,
    default_dataset_path,
    discover_newlayout_checkpoint_specs,
    normalize_model_specs,
    run_mbti_for_model_specs,
)


DEFAULT_MODEL_CONFIGS = {
    "原始基座模型": "./llama-3B-Instruct",
    "F性格模型": "./dpo_outputs/model_f_3B",
    "T性格模型": "./dpo_outputs/model_t_3B",
}


def _parse_model_arg(value: str) -> dict:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use NAME=PATH format")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--model must use NAME=PATH format")
    return {"display_name": name, "checkpoint_path": path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 93-item doubled MBTI test and save per-letter personality scores."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        action="append",
        type=_parse_model_arg,
        default=[],
        help="Model to evaluate, in NAME=PATH format. Can be repeated.",
    )
    source.add_argument(
        "--results-root",
        default=None,
        help="Legacy pipeline results root with pipeline_state.json; model checkpoint paths are read from metadata.",
    )
    source.add_argument(
        "--model-root",
        default=None,
        help="New-layout model root. Only works if base/ and variant folders contain actual checkpoint files.",
    )
    parser.add_argument(
        "--dataset",
        default=str(default_dataset_path()),
        help="Path to MBTI_doubled_93.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="MBTI/results/mbti_types",
        help="Directory for MBTI reports.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of repeated MBTI trials. Logit decoding is deterministic, so 1 is usually enough.",
    )
    parser.add_argument(
        "--decode-method",
        choices=["logit", "generate"],
        default="logit",
        help="Use next-token A/B logit comparison or greedy generation.",
    )
    parser.add_argument(
        "--include-invalid-pairs",
        action="store_true",
        help="Keep cross-dimension JSON items instead of skipping them. Default: skip invalid pairs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if an existing MBTI summary exists for a model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used when loading models.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map argument.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers loaders.",
    )
    return parser


def _collect_model_specs(args: argparse.Namespace) -> list[dict]:
    if args.model:
        return normalize_model_specs(args.model)
    if args.results_root:
        entries = ordered_model_entries(Path(args.results_root))
        return [
            {"display_name": entry["display_name"], "checkpoint_path": entry["checkpoint_path"]}
            for entry in entries
        ]
    if args.model_root:
        return discover_newlayout_checkpoint_specs(Path(args.model_root))

    return normalize_model_specs(DEFAULT_MODEL_CONFIGS.items())


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model_specs = _collect_model_specs(args)
    if not model_specs:
        raise SystemExit("No model checkpoint specs found for MBTI evaluation.")

    config = MBTIRunConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        num_trials=args.num_trials,
        decode_method=args.decode_method,
        skip_invalid_pairs=not args.include_invalid_pairs,
        force=bool(args.force),
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )
    summary_rows = run_mbti_for_model_specs(model_specs, config)
    print(f"[MBTI] Completed. Wrote {len(summary_rows)} model summaries to {config.output_dir}.")


if __name__ == "__main__":
    main()
