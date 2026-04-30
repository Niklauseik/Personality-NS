# -*- coding: utf-8 -*-
"""
Stage-2 pipeline: post-process sentiment outputs, evaluate, and draw charts.

Examples:
  python stage2_process_results.py --results-root results-NS-first-run
  python stage2_process_results.py --results-root results-NS-first-run results-ST-NF-first-run
  python stage2_process_results.py --results-glob "results-*" --continue-on-error

New layout (model root, no pipeline_state.json, ASCII-only paths):
  python stage2_process_results.py --results-root llama-3b_newlayout
"""
from __future__ import annotations

from pathlib import Path

from common.pipeline_utils import get_pipeline_state_path


def _discover_results_roots(patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        roots.extend(sorted(Path.cwd().glob(pattern)))
    return roots


def _is_run_root(path: Path) -> bool:
    return path.is_dir() and get_pipeline_state_path(path).exists()


def _is_newlayout_model_root(path: Path) -> bool:
    return path.is_dir() and (path / "base").is_dir()


def _find_benchmark_root(results_root: Path) -> Path | None:
    """
    Find a benchmark folder associated with the given run root.

    Expected locations:
    - <results_root>/benchmark (legacy + current local runs)
    - <dimension_root>/benchmark (if benchmark stored per dimension)
    - <model_root>/benchmark (if benchmark stored once per model)
    """
    candidates = [
        results_root / "benchmark",
        results_root.parent / "benchmark",
        results_root.parent.parent / "benchmark",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _expand_to_run_roots(target: Path) -> list[Path]:
    """
    Expand an input path into 1+ runnable results roots.

    Supported layouts:
    - Run root (contains pipeline_state.json):
        <anything>/pipeline_state.json
    - New 3-level layout:
        <model_root>/<dimension>/<run_root>/pipeline_state.json
      or <model_root>/<dimension>/<run_root>
    - Compatibility: passing the model_root or dimension folder expands to all run roots below it.
    """
    target = target.expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Results path not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {target}")

    if _is_run_root(target):
        return [target]

    run_roots: list[Path] = []

    # Depth-1: dimension -> run_root, or old patterns where the passed folder contains run_roots directly.
    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        if _is_run_root(child):
            run_roots.append(child)

    if run_roots:
        return run_roots

    # Depth-2: model_root -> dimension -> run_root
    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir():
                continue
            if _is_run_root(grandchild):
                run_roots.append(grandchild)

    return run_roots


def _mbti_config_from_args(args, output_dir: Path):
    from MBTI.code.mbti_eval import MBTIRunConfig, default_dataset_path

    dataset_arg = getattr(args, "mbti_dataset", None)
    dataset_path = Path(dataset_arg) if dataset_arg else default_dataset_path()
    return MBTIRunConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_trials=int(getattr(args, "mbti_trials", 1)),
        decode_method=str(getattr(args, "mbti_decode_method", "logit")),
        skip_invalid_pairs=not bool(getattr(args, "mbti_include_invalid_pairs", False)),
        force=bool(getattr(args, "mbti_force", False)),
        torch_dtype=str(getattr(args, "mbti_torch_dtype", "auto")),
        device_map=str(getattr(args, "mbti_device_map", "auto")),
        trust_remote_code=bool(getattr(args, "mbti_trust_remote_code", False)),
    )


def _run_mbti_for_model_specs(model_specs: list[dict], output_dir: Path, args, context: str) -> None:
    if bool(getattr(args, "skip_mbti", False)):
        print(f"\n[Stage-2] Skipping MBTI validation for {context} (--skip-mbti).")
        return
    if not model_specs:
        print(f"\n[Stage-2] No checkpoint-like model paths found for MBTI validation in {context}; skipping.")
        return

    print(f"\n[Stage-2] Running MBTI personality validation for {context}...")
    from MBTI.code.mbti_eval import run_mbti_for_model_specs

    config = _mbti_config_from_args(args, output_dir=output_dir)
    run_mbti_for_model_specs(model_specs, config)


def _legacy_mbti_model_specs(results_root: Path) -> list[dict]:
    from common.pipeline_utils import ordered_model_entries

    specs: list[dict] = []
    for entry in ordered_model_entries(results_root):
        specs.append(
            {
                "display_name": entry.get("display_name", entry.get("code", "model")),
                "checkpoint_path": entry.get("checkpoint_path", ""),
            }
        )
    return specs


def _newlayout_mbti_model_specs(model_root: Path) -> list[dict]:
    from MBTI.code.mbti_eval import discover_newlayout_checkpoint_specs

    return discover_newlayout_checkpoint_specs(model_root)


def _run_stage2_for_root(results_root: Path, args=None) -> None:
    print(f"\n========== [Stage-2] Processing: {results_root} ==========")
    if args is not None:
        _run_mbti_for_model_specs(
            _legacy_mbti_model_specs(results_root),
            output_dir=results_root / "mbti",
            args=args,
            context=str(results_root),
        )
    print("\n[Stage-2] Collecting invalid sentiment predictions...")
    from .sentiment_get_invalid import collect_invalid_predictions

    collect_invalid_predictions(results_root)
    print("\n[Stage-2] Correcting invalid sentiment predictions...")
    try:
        from .sentiment_label_correct import process_all as correct_invalid_sentiments
    except Exception as exc:  # pragma: no cover
        correct_invalid_sentiments = None
        print(f"[Stage-2] NOTE: sentiment_label_correct unavailable; skipping GPT correction. ({exc})")
    if correct_invalid_sentiments is not None:
        correct_invalid_sentiments(results_root)
    print("\n[Stage-2] Merging corrected labels...")
    from .sentiment_label_merge import merge_corrected_labels

    merge_corrected_labels(results_root)
    print("\n[Stage-2] Summarizing label distributions...")
    from .sentiment_label_count import summarize_label_distribution

    chart_data = summarize_label_distribution(results_root)
    print("\n[Stage-2] Evaluating sentiment performance...")
    from .evaluate_sentiment import evaluate_sentiment

    evaluate_sentiment(results_root)
    print("\n[Stage-2] Evaluating benchmark performance...")
    from .evaluate_benchmarks import evaluate_benchmarks

    benchmark_root = _find_benchmark_root(results_root)
    if benchmark_root is not None:
        evaluate_benchmarks(results_root, benchmark_root=benchmark_root)
    else:
        print(
            "[Stage-2] Benchmark results not found under "
            f"{results_root / 'benchmark'} (or parent folders); skipping."
        )
    print("\n[Stage-2] Generating visualizations...")
    from .draw_charts import generate_charts

    generate_charts(results_root, chart_data)
    print("\n[Stage-2] Completed all processing steps.")


def _run_stage2_for_newlayout_model_root(model_root: Path, args=None) -> None:
    print(f"\n========== [Stage-2] Processing model root (new layout): {model_root} ==========")
    if args is not None:
        _run_mbti_for_model_specs(
            _newlayout_mbti_model_specs(model_root),
            output_dir=model_root / "mbti",
            args=args,
            context=str(model_root),
        )
    from .newlayout import process_model_root

    process_model_root(model_root)
    print("\n[Stage-2] Completed new-layout processing.")


def _unique_existing_roots(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def run(args) -> None:
    requested = [Path(p) for p in args.results_root] if args.results_root else []
    discovered = _discover_results_roots(args.results_glob)
    targets = _unique_existing_roots(requested + discovered)
    if not targets:
        targets = [Path("results")]

    if args.dry_run:
        print("[Stage-2] Dry-run. Would process:")
        for target in targets:
            if _is_newlayout_model_root(target):
                print(f"  - {target} (new layout model root)")
            else:
                expanded = _expand_to_run_roots(target)
                if not expanded:
                    print(f"  - {target} (no run roots found)")
                    continue
                for root in expanded:
                    print(f"  - {root}")
        return

    legacy_run_roots: list[Path] = []
    newlayout_model_roots: list[Path] = []
    for target in targets:
        if _is_newlayout_model_root(target):
            newlayout_model_roots.append(target)
            continue

        expanded = _expand_to_run_roots(target)
        if not expanded:
            raise FileNotFoundError(
                f"Unrecognized results root: {target}. "
                "Expected either a new-layout model root (contains 'base/'), "
                "or a legacy run root containing pipeline_state.json, "
                "or a folder containing legacy run roots."
            )
        legacy_run_roots.extend(expanded)

    legacy_run_roots = _unique_existing_roots(legacy_run_roots)
    newlayout_model_roots = _unique_existing_roots(newlayout_model_roots)

    failures = 0
    for root in newlayout_model_roots:
        try:
            _run_stage2_for_newlayout_model_root(root, args=args)
        except Exception as exc:
            failures += 1
            print(f"\n[Stage-2] ERROR while processing new layout {root}: {exc}")
            if not args.continue_on_error:
                raise

    for root in legacy_run_roots:
        if not _is_run_root(root):
            raise FileNotFoundError(f"Run root missing pipeline_state.json: {root}")
        try:
            _run_stage2_for_root(root, args=args)
        except Exception as exc:
            failures += 1
            print(f"\n[Stage-2] ERROR while processing legacy run root {root}: {exc}")
            if not args.continue_on_error:
                raise
    if failures:
        raise SystemExit(f"[Stage-2] Completed with failures: {failures}")
