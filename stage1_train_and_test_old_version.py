# -*- coding: utf-8 -*-
"""
Stage-1 pipeline (old version): train requested personality models
using the CSV-based DPO format, then (optionally) run benchmark and
sentiment inference.
"""
import argparse
import re
import shutil
from pathlib import Path
from typing import List

from dpo_training_old_version import train_personality_model
from pipeline_utils import (
    build_model_entries,
    current_timestamp,
    ensure_output_target,
    generate_run_id,
    get_dimension_spec,
    normalize_path,
    resolve_letter_spec,
    sanitize_run_name,
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
    return parser.parse_args()


def _train_all_subtypes(dimension: str, base_model_path: Path, output_root: Path,
                        ) -> List[dict]:
    spec = get_dimension_spec(dimension)
    trained = []
    for subtype in spec["subtypes"]:
        target_dir = standard_model_dir(output_root, base_model_path, subtype["code"][0].upper())
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
    target_dir = standard_model_dir(output_root, base_model_path, normalized)
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


_RUN_SUFFIX_RE = re.compile(r"(?i)-run\d+$")


def _strip_run_suffix(path: Path) -> Path:
    name = path.name
    if _RUN_SUFFIX_RE.search(name):
        return path.with_name(_RUN_SUFFIX_RE.sub("", name))
    return path


def _derive_results_prefix(args, base_model_path: Path) -> Path:
    # Explicit override wins (treated as a prefix; any trailing "-runX" is stripped).
    if args.results_root:
        return _strip_run_suffix(normalize_path(args.results_root))

    base_name = sanitize_run_name(base_model_path.name)
    if args.pair:
        codes = [_normalize_personality_code(code) for code in args.pair]
        pair_tag = "-".join(codes)
    else:
        spec = get_dimension_spec(args.dimension)
        letters = sorted(subtype["code"][0].upper() for subtype in spec["subtypes"])
        pair_tag = "-".join(letters)
    pair_tag = sanitize_run_name(pair_tag)
    return normalize_path(f"results-{base_name}-{pair_tag}")


def _results_root_for_run(prefix: Path, run_idx: int) -> Path:
    return prefix.with_name(f"{prefix.name}-run{run_idx}")


def main():
    args = _parse_args()
    base_model_path = normalize_path(args.model_path)
    output_root = normalize_path(args.output_root)
    results_prefix = _derive_results_prefix(args, base_model_path)
    if args.sentiment_runs < 1:
        raise ValueError("--sentiment-runs must be >= 1")
    results_roots = [_results_root_for_run(results_prefix, i) for i in range(1, args.sentiment_runs + 1)]
    print("\n[Stage-1] Results will be stored under:")
    for root in results_roots:
        print(f"  - {root}")

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
    sentiment_model_specs = [
        {"display_name": entry["display_name"], "checkpoint_path": entry["checkpoint_path"]}
        for entry in model_entries
        if args.base_sentiment or entry.get("role") != "base"
    ]

    session_id = generate_run_id()
    timestamp = current_timestamp()
    metadata_paths: list[Path] = []
    for run_idx, results_root in enumerate(results_roots, start=1):
        state = {
            "run_id": f"{session_id}-run{run_idx}",
            "session_id": session_id,
            "dimension": args.dimension,
            "pair": [code.upper() for code in args.pair] if args.pair else None,
            "timestamp": timestamp,
            "results_root": str(results_root),
            "results_root_prefix": str(results_prefix),
            "output_root": str(output_root),
            "base_model_path": str(base_model_path),
            "benchmark_enabled": bool(args.benchmark),
            "base_sentiment_enabled": bool(args.base_sentiment),
            "sentiment_runs": 1,
            "sentiment_runs_total": int(args.sentiment_runs),
            "sentiment_run_index": int(run_idx),
            "model_entries": model_entries,
        }
        metadata_paths.append(write_pipeline_state(state, results_root))

    if args.benchmark:
        print("\n[Stage-1] Running benchmark evaluations...")
        benchmark_root = results_roots[0]
        run_benchmarks(model_specs, results_root=benchmark_root)
        if len(results_roots) > 1:
            src = benchmark_root / "benchmark"
            if src.exists():
                for dst_root in results_roots[1:]:
                    dst = dst_root / "benchmark"
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                print(f"[Stage-1] Benchmark output not found at {src}; skipping copy to other runs.")
    else:
        print("\n[Stage-1] Benchmark is disabled; skipping benchmark evaluations.")
    for run_idx, results_root in enumerate(results_roots, start=1):
        banner = f" (run {run_idx}/{len(results_roots)})" if len(results_roots) > 1 else ""
        print(f"\n[Stage-1] Running sentiment inference{banner}...")
        run_sentiment(sentiment_model_specs, results_root=results_root, file_suffix="")

    print("\n[Stage-1] Completed. Metadata saved to:")
    for path in metadata_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
