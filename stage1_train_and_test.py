# -*- coding: utf-8 -*-
"""
Stage-1 pipeline: train the requested dimension models,
then run benchmark and sentiment inference.

python stage1_train_and_test.py --dimension information information --model-path ./llama-3B-Instruct

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
    standard_model_dir,
    write_pipeline_state,
)
from run_benchmark import run_benchmarks
from run_sentiment import run_sentiment


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage-1: Train models and generate raw benchmark/sentiment results.")
    parser.add_argument("--dimension", required=True, choices=["energy", "information", "decision", "execution"])
    parser.add_argument("--model-path", required=True, help="Base model checkpoint path.")
    parser.add_argument("--output-root", default="dpo_outputs", help="Directory to store trained checkpoints.")
    parser.add_argument("--results-root", default="results", help="Directory where evaluation results are stored.")
    parser.add_argument("--take-n", type=int, default=10000, help="Limit DPO training samples.")
    parser.add_argument("--overwrite-models", action="store_true", help="Allow overwriting existing checkpoints.")
    return parser.parse_args()


def _train_all_subtypes(dimension: str, base_model_path: Path, output_root: Path,
                        take_n: int, overwrite: bool) -> List[dict]:
    spec = get_dimension_spec(dimension)
    trained = []
    for subtype in spec["subtypes"]:
        target_dir = standard_model_dir(output_root, subtype["code"])
        ensure_output_target(target_dir, overwrite=overwrite)
        print(f"\n🚀 Training {subtype['display_name']} -> {target_dir}")
        train_personality_model(
            dimension=dimension,
            preferred_subtype=subtype["preferred"],
            base_model_path=str(base_model_path),
            save_path=str(target_dir),
            take_n=take_n,
        )
        trained.append({
            "role": "trained",
            "code": subtype["code"],
            "subtype": subtype["preferred"],
            "display_name": subtype["display_name"],
            "checkpoint_path": str(target_dir),
        })
    return trained


def main():
    args = _parse_args()
    base_model_path = normalize_path(args.model_path)
    output_root = normalize_path(args.output_root)
    results_root = normalize_path(args.results_root)

    print("\n[Stage-1] Training personality subtypes...")
    trained_models = _train_all_subtypes(
        dimension=args.dimension,
        base_model_path=base_model_path,
        output_root=output_root,
        take_n=args.take_n,
        overwrite=args.overwrite_models,
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
