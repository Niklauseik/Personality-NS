# -*- coding: utf-8 -*-
"""
Stage-1 pipeline: train requested personality models,
then run benchmark and sentiment inference.

Examples:
  python stage1_train_and_test.py --dimension information --model-path ./llama-3B-Instruct
  python stage1_train_and_test.py --pair ENTP ISFJ --model-path ./llama-3B-Instruct
"""
import argparse
from pathlib import Path
from typing import List

from dpo_training_chat import train_personality_model
from pipeline_utils import (
    build_model_entries,
    current_timestamp,
    ensure_output_target,
    generate_run_id,
    get_dimension_spec,
    normalize_path,
    resolve_letter_spec,
    standard_model_dir,
    write_pipeline_state,
)
from run_benchmark import run_benchmarks
from run_sentiment import run_sentiment


def _parse_args():
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
    parser.add_argument("--results-root", default="results", help="Directory where evaluation results are stored.")
    return parser.parse_args()


def _train_all_subtypes(dimension: str, base_model_path: Path, output_root: Path,
                        ) -> List[dict]:
    spec = get_dimension_spec(dimension)
    trained = []
    for subtype in spec["subtypes"]:
        target_dir = standard_model_dir(output_root, subtype["code"])
        ensure_output_target(target_dir)
        print(f"\n🚀 Training {subtype['display_name']} -> {target_dir}")
        train_personality_model(
            dimension=dimension,
            preferred_subtype=subtype["preferred"],
            base_model_path=str(base_model_path),
            save_path=str(target_dir),
        )
        trained.append({
            "role": "trained",
            "code": subtype["code"],
            "subtype": subtype["preferred"],
            "display_name": subtype["display_name"],
            "checkpoint_path": str(target_dir),
        })
    return trained


def _normalize_personality_code(raw_code: str) -> str:
    normalized = (raw_code or "").strip().upper()
    if not normalized:
        raise ValueError("Personality code cannot be empty.")
    return normalized


def _build_training_sequence(code: str) -> List[dict]:
    seen_dimensions = set()
    sequence = []
    for letter in code:
        step = resolve_letter_spec(letter)
        dimension = step["dimension"]
        if dimension in seen_dimensions:
            raise ValueError(
                f"Duplicate dimension detected in '{code}'. "
                "Each dimension (energy, information, decision, execution) can appear at most once."
            )
        seen_dimensions.add(dimension)
        sequence.append({"letter": letter, "dimension": dimension, "preferred": step["preferred"]})
    return sequence


def _train_personality_code(code: str, base_model_path: Path, output_root: Path,
                            ) -> dict:
    normalized = _normalize_personality_code(code)
    sequence = _build_training_sequence(normalized)
    target_dir = standard_model_dir(output_root, normalized.lower())
    ensure_output_target(target_dir)

    print(f"\n🚀 Training {normalized} (letters: {'-'.join(step['letter'] for step in sequence)}) -> {target_dir}")
    train_personality_model(
        dimension=None,
        preferred_subtype=None,
        base_model_path=str(base_model_path),
        save_path=str(target_dir),
        personality_sequence=sequence,
    )

    return {
        "role": "trained",
        "code": normalized.lower(),
        "subtype": normalized,
        "display_name": f"{normalized}性格模型",
        "checkpoint_path": str(target_dir),
        "personality_sequence": sequence,
    }


def _train_personality_pair(pair: List[str], base_model_path: Path, output_root: Path,
                            ) -> List[dict]:
    trained = []
    for raw_code in pair:
        trained.append(
            _train_personality_code(
                code=raw_code,
                base_model_path=base_model_path,
                output_root=output_root,
            )
        )
    return trained


def main():
    args = _parse_args()
    base_model_path = normalize_path(args.model_path)
    output_root = normalize_path(args.output_root)
    results_root = normalize_path(args.results_root)

    if args.dimension:
        print("\n[Stage-1] Training personality subtypes...")
        trained_models = _train_all_subtypes(
            dimension=args.dimension,
            base_model_path=base_model_path,
            output_root=output_root,
        )
    else:
        pair_display = " vs ".join(_normalize_personality_code(code) for code in args.pair)
        print(f"\n[Stage-1] Training custom personality pair: {pair_display}")
        trained_models = _train_personality_pair(
            pair=args.pair,
            base_model_path=base_model_path,
            output_root=output_root,
        )

    model_entries = build_model_entries(
        dimension=args.dimension,
        base_model_path=base_model_path,
        trained_models=trained_models,
    )
    model_specs = [
        {"display_name": entry["display_name"], "checkpoint_path": entry["checkpoint_path"]}
        for entry in model_entries
    ]

    print("\n[Stage-1] Running benchmark evaluations...")
    run_benchmarks(model_specs, results_root=results_root)
    print("\n[Stage-1] Running sentiment inference...")
    run_sentiment(model_specs, results_root=results_root)

    state = {
        "run_id": generate_run_id(),
        "dimension": args.dimension,
        "pair": [code.upper() for code in args.pair] if args.pair else None,
        "timestamp": current_timestamp(),
        "results_root": args.results_root,
        "output_root": args.output_root,
        "base_model_path": args.model_path,
        "model_entries": model_entries,
    }
    metadata_path = write_pipeline_state(state, results_root)
    print(f"\n[Stage-1] Completed. Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
