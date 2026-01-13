# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import ordered_sentiment_entries, resolve_dataset_base

DATASETS = [
    {"name": "imdb_sentiment", "file": "imdb_sentiment_results.csv",
     "label_map": {"0": "positive", "1": "negative"},
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


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())


def build_allowed(ds) -> list:
    if ds["allowed_labels"]:
        allowed = [clean(x) for x in ds["allowed_labels"]]
    elif ds["label_map"]:
        allowed = [clean(x) for x in ds["label_map"].values()]
    else:
        allowed = []
    return sorted(set(allowed))


def map_true_label_series(ds, series: pd.Series) -> pd.Series:
    if ds["label_map"]:
        series = series.astype(str).map(ds["label_map"])
    return series.astype(str).apply(clean)


def extract_pred_label(text: str, allowed: list) -> str:
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    text_l = text.lower()
    earliest, pos = None, 10**12
    for lbl in allowed:
        match = re.search(rf"\b{re.escape(lbl)}\b", text_l)
        if match and match.start() < pos:
            earliest, pos = lbl, match.start()
    return earliest if earliest is not None else "invalid"


def compute_metrics(y_true, y_pred, class_labels):
    y_true_seq = list(y_true)
    y_pred_seq = list(y_pred)
    labels = list(class_labels)

    acc = (sum(t == p for t, p in zip(y_true_seq, y_pred_seq)) / len(y_true_seq)) if y_true_seq else 0.0

    stats = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}
    for t, p in zip(y_true_seq, y_pred_seq):
        if t in stats:
            stats[t]["support"] += 1
        if t == p and t in stats:
            stats[t]["tp"] += 1
            continue
        if t in stats:
            stats[t]["fn"] += 1
        if p in stats:
            stats[p]["fp"] += 1

    per_label = []
    for lbl in labels:
        tp = stats[lbl]["tp"]
        fp = stats[lbl]["fp"]
        fn = stats[lbl]["fn"]
        support = stats[lbl]["support"]
        precision_l = tp / (tp + fp) if (tp + fp) else 0.0
        recall_l = tp / (tp + fn) if (tp + fn) else 0.0
        f1_l = (2 * precision_l * recall_l / (precision_l + recall_l)) if (precision_l + recall_l) else 0.0
        per_label.append(
            {
                "precision": precision_l,
                "recall": recall_l,
                "f1": f1_l,
                "support": support,
            }
        )

    if per_label:
        p_m = sum(x["precision"] for x in per_label) / len(per_label)
        r_m = sum(x["recall"] for x in per_label) / len(per_label)
        f_m = sum(x["f1"] for x in per_label) / len(per_label)

        support_total = sum(x["support"] for x in per_label)
        if support_total:
            p_w = sum(x["precision"] * x["support"] for x in per_label) / support_total
            r_w = sum(x["recall"] * x["support"] for x in per_label) / support_total
            f_w = sum(x["f1"] * x["support"] for x in per_label) / support_total
        else:
            p_w = r_w = f_w = 0.0
    else:
        p_m = r_m = f_m = 0.0
        p_w = r_w = f_w = 0.0
    return {
        "accuracy": acc,
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "support": int(len(y_true)),
    }


def pick_pred_path(base_path: Path) -> Path | None:
    relabeled = base_path.with_suffix(".relabeled.csv")
    if relabeled.exists():
        return relabeled
    return base_path if base_path.exists() else None


def evaluate_sentiment(results_root: Path | str = "results"):
    results_root = Path(results_root)
    entries = ordered_sentiment_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")

    rows = []

    for ds in DATASETS:
        print(f"🔍 处理数据集：{ds['name']}")
        allowed = build_allowed(ds)
        if not allowed:
            print(f"  ⚠️ 数据集 {ds['name']} 未能解析到合法标签集合，跳过。")
            continue

        base_dir = resolve_dataset_base(results_root, ds["base_path"])

        for entry in entries:
            model_folder = entry["display_name"]
            path = pick_pred_path(base_dir / model_folder / ds["file"])
            if not path:
                print(f"  ⚠️ 缺少文件：{base_dir / model_folder / ds['file']}")
                continue

            df = pd.read_csv(path)

            y_true_all = map_true_label_series(ds, df[ds["label_col"]])
            mask_keep = y_true_all.isin(allowed)
            kept = df[mask_keep].copy()
            if kept.empty:
                print(f"  ⚠️ {ds['name']} - {model_folder}: 无可评估样本。")
                continue

            kept["__pred_raw"] = kept[ds["pred_col"]].astype(str)
            kept["__pred_label"] = kept["__pred_raw"].apply(lambda x: extract_pred_label(x, allowed))

            y_true = map_true_label_series(ds, kept[ds["label_col"]]).tolist()
            y_pred = kept["__pred_label"].tolist()

            metrics = compute_metrics(y_true, y_pred, class_labels=allowed)

            rows.append({
                "dataset": ds["name"],
                "model_code": entry["code"],
                "model_display": model_folder,
                "labels": "|".join(allowed),
                **metrics
            })

    if rows:
        out_df = pd.DataFrame(rows)
        col_order = [
            "dataset", "model_code", "model_display", "labels", "support",
            "accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ]
        out_df = out_df[col_order]

        num_cols = [
            "accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ]
        out_df[num_cols] = out_df[num_cols].round(2)

        out_dir = results_root / "sentiment"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "metrics_summary.csv"
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("\n✅ 指标计算完成。")
        print(f"  - CSV: {csv_path}")
    else:
        print("⚠️ 未生成任何指标结果，请检查文件路径与数据。")


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment results for all active models.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_sentiment(results_root=Path(args.results_root))
