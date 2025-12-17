# -*- coding: utf-8 -*-
"""
Stage-2 pipeline: post-process sentiment outputs, evaluate, and draw charts.

Examples:
  python stage2_process_results.py --results-root results-NS-first-run
  python stage2_process_results.py --results-root results-NS-first-run results-ST-NF-first-run
  python stage2_process_results.py --results-glob "results-*" --continue-on-error
"""
import argparse
from pathlib import Path

from draw_charts import generate_charts
from evaluate_benchmarks import evaluate_benchmarks
from evaluate_sentiment import evaluate_sentiment
from sentiment_get_invalid import collect_invalid_predictions
from sentiment_label_correct import process_all as correct_invalid_sentiments
from sentiment_label_count import summarize_label_distribution
from sentiment_label_merge import merge_corrected_labels


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


def _run_stage2_for_root(results_root: Path) -> None:
    print(f"\n========== [Stage-2] Processing: {results_root} ==========")
    print("\n[Stage-2] Collecting invalid sentiment predictions...")
    collect_invalid_predictions(results_root)
    print("\n[Stage-2] Correcting invalid sentiment predictions...")
    correct_invalid_sentiments(results_root)
    print("\n[Stage-2] Merging corrected labels...")
    merge_corrected_labels(results_root)
    print("\n[Stage-2] Summarizing label distributions...")
    chart_data = summarize_label_distribution(results_root)
    print("\n[Stage-2] Evaluating sentiment performance...")
    evaluate_sentiment(results_root)
    print("\n[Stage-2] Evaluating benchmark performance...")
    if (results_root / "benchmark").exists():
        evaluate_benchmarks(results_root)
    else:
        print(f"[Stage-2] Benchmark results not found under {results_root / 'benchmark'}; skipping.")
    print("\n[Stage-2] Generating visualizations...")
    generate_charts(results_root, chart_data)
    print("\n[Stage-2] Completed all processing steps.")


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
    roots = _unique_existing_roots(requested + discovered)
    if not roots:
        roots = [Path("results")]

    if args.dry_run:
        print("[Stage-2] Dry-run. Would process:")
        for root in roots:
            print(f"  - {root}")
        return

    failures = 0
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Results root not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Results root is not a directory: {root}")
        try:
            _run_stage2_for_root(root)
        except Exception as exc:
            failures += 1
            print(f"\n[Stage-2] ERROR while processing {root}: {exc}")
            if not args.continue_on_error:
                raise
    if failures:
        raise SystemExit(f"[Stage-2] Completed with failures: {failures}")


if __name__ == "__main__":
    main()
