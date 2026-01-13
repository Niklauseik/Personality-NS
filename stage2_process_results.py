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
import argparse
from pathlib import Path

from pipeline_utils import get_pipeline_state_path


def _parse_args():
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
    return parser.parse_args()


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


def _run_stage2_for_root(results_root: Path) -> None:
    print(f"\n========== [Stage-2] Processing: {results_root} ==========")
    print("\n[Stage-2] Collecting invalid sentiment predictions...")
    from sentiment_get_invalid import collect_invalid_predictions

    collect_invalid_predictions(results_root)
    print("\n[Stage-2] Correcting invalid sentiment predictions...")
    try:
        from sentiment_label_correct import process_all as correct_invalid_sentiments
    except Exception as exc:  # pragma: no cover
        correct_invalid_sentiments = None
        print(f"[Stage-2] NOTE: sentiment_label_correct unavailable; skipping GPT correction. ({exc})")
    if correct_invalid_sentiments is not None:
        correct_invalid_sentiments(results_root)
    print("\n[Stage-2] Merging corrected labels...")
    from sentiment_label_merge import merge_corrected_labels

    merge_corrected_labels(results_root)
    print("\n[Stage-2] Summarizing label distributions...")
    from sentiment_label_count import summarize_label_distribution

    chart_data = summarize_label_distribution(results_root)
    print("\n[Stage-2] Evaluating sentiment performance...")
    from evaluate_sentiment import evaluate_sentiment

    evaluate_sentiment(results_root)
    print("\n[Stage-2] Evaluating benchmark performance...")
    from evaluate_benchmarks import evaluate_benchmarks

    benchmark_root = _find_benchmark_root(results_root)
    if benchmark_root is not None:
        evaluate_benchmarks(results_root, benchmark_root=benchmark_root)
    else:
        print(
            "[Stage-2] Benchmark results not found under "
            f"{results_root / 'benchmark'} (or parent folders); skipping."
        )
    print("\n[Stage-2] Generating visualizations...")
    from draw_charts import generate_charts

    generate_charts(results_root, chart_data)
    print("\n[Stage-2] Completed all processing steps.")


def _run_stage2_for_newlayout_model_root(model_root: Path) -> None:
    print(f"\n========== [Stage-2] Processing model root (new layout): {model_root} ==========")
    from stage2_newlayout import process_model_root

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


def main():
    args = _parse_args()
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
            _run_stage2_for_newlayout_model_root(root)
        except Exception as exc:
            failures += 1
            print(f"\n[Stage-2] ERROR while processing new layout {root}: {exc}")
            if not args.continue_on_error:
                raise

    for root in legacy_run_roots:
        if not _is_run_root(root):
            raise FileNotFoundError(f"Run root missing pipeline_state.json: {root}")
        try:
            _run_stage2_for_root(root)
        except Exception as exc:
            failures += 1
            print(f"\n[Stage-2] ERROR while processing legacy run root {root}: {exc}")
            if not args.continue_on_error:
                raise
    if failures:
        raise SystemExit(f"[Stage-2] Completed with failures: {failures}")


if __name__ == "__main__":
    main()
