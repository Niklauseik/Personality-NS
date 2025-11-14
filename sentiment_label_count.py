# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline_utils import ordered_model_entries, resolve_dataset_base

DATASETS = [
    {"name": "imdb_sentiment", "file": "imdb_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None, "label_col": "label", "pred_col": "prediction",
     "base_path": "results/sentiment/imdb"},
    {"name": "mental_sentiment", "file": "mental_sentiment_results.csv",
     "label_map": None, "allowed_labels": ["normal", "depression"],
     "label_col": "label", "pred_col": "prediction",
     "base_path": "results/sentiment/mental"},
    {"name": "news_sentiment", "file": "news_sentiment_results.csv",
     "label_map": {"0": "bearish", "1": "bullish", "2": "neutral"},
     "allowed_labels": None, "label_col": "label", "pred_col": "prediction",
     "base_path": "results/sentiment/news"},
    {"name": "fiqasa_sentiment", "file": "fiqasa_sentiment_results.csv",
     "label_map": None, "allowed_labels": ["negative", "positive", "neutral"],
     "label_col": "answer", "pred_col": "prediction",
     "base_path": "results/sentiment/fiqasa"},
    {"name": "imdb_sklearn", "file": "imdb_sklearn_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None, "label_col": "label", "pred_col": "prediction",
     "base_path": "results/sentiment/imdb_sklearn"},
    {"name": "sst2", "file": "sst2_sentiment_results.csv",
     "label_map": {"0": "negative", "1": "positive"},
     "allowed_labels": None, "label_col": "label", "pred_col": "prediction",
     "base_path": "results/sentiment/sst2"},
]

SUMMARY_TXT = "label_distribution_summary.txt"
TRUE_COL = "真实数量"


def norm_token(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().rstrip(".。…")


def determine_true_labels(ds_cfg: dict, df_true: pd.DataFrame) -> List[str]:
    if ds_cfg["allowed_labels"]:
        return sorted({norm_token(x) for x in ds_cfg["allowed_labels"]})
    if ds_cfg["label_map"]:
        return sorted({norm_token(v) for v in ds_cfg["label_map"].values()})
    return sorted({norm_token(x) for x in df_true[ds_cfg["label_col"]].astype(str)})


def pick_pred_path(base_path: Path) -> Path | None:
    relabeled = base_path.with_suffix(".relabeled.csv")
    processed = base_path.with_suffix(".processed.csv")
    for candidate in (relabeled, processed, base_path):
        if candidate.exists():
            return candidate
    return None


def classify_prediction(pred_text: str, valid_pred_set: set[str]) -> str:
    token = norm_token(pred_text)
    return token if token in valid_pred_set else "invalid"


def summarize_label_distribution(results_root: Path | str = "results"):
    results_root = Path(results_root)
    entries = ordered_model_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")
    model_names = [entry["display_name"] for entry in entries]

    dist_all = defaultdict(lambda: defaultdict(lambda: {TRUE_COL: 0, **{name: 0 for name in model_names}}))
    label_order_map: Dict[str, List[str]] = {}

    for ds in DATASETS:
        print(f"[INFO] Processing dataset: {ds['name']}")
        base_dir = resolve_dataset_base(results_root, ds["base_path"])

        # Locate a source file for true labels.
        true_source = None
        for model_dir in model_names:
            candidate = base_dir / model_dir / ds["file"]
            if candidate.exists():
                true_source = candidate
                break
        if not true_source:
            print(f"  [WARN] Missing source file for labels under: {base_dir}")
            continue

        df_true = pd.read_csv(true_source, dtype=str).fillna("")
        if ds["label_map"]:
            df_true[ds["label_col"]] = df_true[ds["label_col"]].astype(str).map(ds["label_map"]).fillna("")
        df_true[ds["label_col"]] = df_true[ds["label_col"]].astype(str).map(norm_token)

        true_labels = determine_true_labels(ds, df_true)
        ordered = true_labels + [x for x in ["neutral", "mixed", "invalid"] if x not in true_labels]
        valid_pred_set = set(ordered)
        label_order_map[ds["name"]] = ordered

        counts_true = Counter(df_true[ds["label_col"]])
        for lbl in true_labels:
            dist_all[ds["name"]][lbl][TRUE_COL] = int(counts_true.get(lbl, 0))
        for extra_lbl in ["neutral", "mixed", "invalid"]:
            dist_all[ds["name"]][extra_lbl][TRUE_COL] = dist_all[ds["name"]][extra_lbl].get(TRUE_COL, 0)

        for model_name in model_names:
            raw_base = base_dir / model_name / ds["file"]
            pred_path = pick_pred_path(raw_base)
            if not pred_path:
                print(f"  [WARN] Missing predictions for: {raw_base}")
                continue

            df_pred = pd.read_csv(pred_path, dtype=str).fillna("")
            for pred in df_pred[ds["pred_col"]].astype(str):
                category = classify_prediction(pred, valid_pred_set)
                dist_all[ds["name"]][category][model_name] += 1

    chart_data = write_summary_txt(dist_all, label_order_map, model_names)
    return chart_data


def write_summary_txt(dist_all, label_order_map, model_names):
    chart_data: Dict[str, Dict] = {}
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        for dataset, label_dict in dist_all.items():
            if dataset not in label_order_map:
                continue
            order = label_order_map[dataset]
            df_out = (
                pd.DataFrame(label_dict).T
                .fillna(0).astype(int)
                .loc[order, [TRUE_COL] + model_names]
            )
            f.write(f"======== {dataset} ========\n")
            f.write(df_out.to_string())
            f.write("\n\n")

            chart_data[dataset] = {
                "labels": order,
                "models": {
                    model_name: {label: int(label_dict[label].get(model_name, 0)) for label in order}
                    for model_name in model_names
                },
            }

    print(f"\n[INFO] Summary saved to {SUMMARY_TXT}")
    return chart_data


if __name__ == "__main__":
    summarize_label_distribution()
