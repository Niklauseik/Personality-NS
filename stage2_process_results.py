# -*- coding: utf-8 -*-
"""
Stage-2 pipeline: post-process sentiment outputs, evaluate, and draw charts.
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
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


def main():
    args = _parse_args()
    results_root = Path(args.results_root)

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
    evaluate_benchmarks(results_root)
    print("\n[Stage-2] Generating visualizations...")
    generate_charts(results_root, chart_data)

    print("\n[Stage-2] Completed all processing steps.")


if __name__ == "__main__":
    main()
