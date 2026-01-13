from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import ordered_model_entries

FILES = {
    "ARC (easy)": "arc_easy_test800_results.csv",
    "BoolQ": "boolq_train800_results.csv",
    "GSM8K": "gsm8k_test800_results.csv",
}


def extract_upper_letter(text):
    match = re.search(r"\b([A-D])\b", str(text).upper())
    return match.group(1) if match else None


def extract_bool(text):
    if isinstance(text, str):
        text_lower = text.lower()
        if "true" in text_lower:
            return True
        if "false" in text_lower:
            return False
    elif isinstance(text, bool):
        return text
    return None


def compute_metrics(y_true, y_pred):
    y_true_seq = list(y_true)
    y_pred_seq = list(y_pred)

    accuracy = sum(t == p for t, p in zip(y_true_seq, y_pred_seq)) / len(y_true_seq) if y_true_seq else 0.0
    labels = []
    seen = set()
    for value in y_true_seq:
        if value in seen:
            continue
        seen.add(value)
        labels.append(value)

    per_label = []
    for label in labels:
        tp = sum((t == label) and (p == label) for t, p in zip(y_true_seq, y_pred_seq))
        fp = sum((t != label) and (p == label) for t, p in zip(y_true_seq, y_pred_seq))
        fn = sum((t == label) and (p != label) for t, p in zip(y_true_seq, y_pred_seq))
        precision_l = tp / (tp + fp) if (tp + fp) else 0.0
        recall_l = tp / (tp + fn) if (tp + fn) else 0.0
        f1_l = (2 * precision_l * recall_l / (precision_l + recall_l)) if (precision_l + recall_l) else 0.0
        per_label.append((precision_l, recall_l, f1_l))

    if per_label:
        precision = sum(p for p, _, _ in per_label) / len(per_label)
        recall = sum(r for _, r, _ in per_label) / len(per_label)
        f1 = sum(f for _, _, f in per_label) / len(per_label)
    else:
        precision = recall = f1 = 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extract_numbers(text):
    text = str(text).replace(",", "").replace("$", "")
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


def gsm8k_accuracy_from_numbers(df):
    correct, total = 0, 0
    for _, row in df.iterrows():
        label_nums = extract_numbers(row["label"])
        pred_nums = extract_numbers(row["prediction"])
        if not label_nums or not pred_nums:
            continue
        label = label_nums[0]
        if label in pred_nums:
            correct += 1
        total += 1
    acc = correct / total if total else 0.0
    return round(acc, 4), correct, total


def evaluate_benchmarks(results_root: Path | str = "results", benchmark_root: Path | str | None = None):
    results_root = Path(results_root)
    entries = ordered_model_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")
    model_dirs = [entry["display_name"] for entry in entries]

    base_path = Path(benchmark_root) if benchmark_root is not None else (results_root / "benchmark")
    if not base_path.exists():
        print(f"[WARN] Benchmark folder not found, skipping: {base_path}")
        return

    all_results = []

    for model_name in model_dirs:
        model_path = base_path / model_name

        for dataset_name, filename in FILES.items():
            file_path = model_path / filename
            if not file_path.exists():
                print(f"[WARN] Missing benchmark file: {file_path}")
                continue

            df = pd.read_csv(file_path)

            if dataset_name == "ARC (easy)":
                df["label_clean"] = df["label"].apply(extract_upper_letter)
                df["prediction_clean"] = df["prediction"].apply(extract_upper_letter)
                df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
                metrics = compute_metrics(df_valid["label_clean"], df_valid["prediction_clean"])

            elif dataset_name == "BoolQ":
                df["label_clean"] = df["label"].apply(extract_bool)
                df["prediction_clean"] = df["prediction"].apply(extract_bool)
                df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
                metrics = compute_metrics(df_valid["label_clean"], df_valid["prediction_clean"])

            elif dataset_name == "GSM8K":
                accuracy, correct, total = gsm8k_accuracy_from_numbers(df)
                metrics = {
                    "accuracy": accuracy,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "correct": correct,
                    "total": total,
                }
            else:
                metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}

            all_results.append(
                {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    **metrics,
                }
            )

    df_metrics = pd.DataFrame(all_results)
    if df_metrics.empty:
        print(f"[WARN] No benchmark result files found under: {base_path} (skipping summary)")
        return

    output_path = base_path / "benchmark_metrics_summary.csv"
    df_metrics.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results for all active models.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_benchmarks(results_root=Path(args.results_root))
