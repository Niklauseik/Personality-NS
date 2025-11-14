# -*- coding: utf-8 -*-
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from pipeline_utils import ordered_model_entries, resolve_dataset_base

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
    import numpy as np
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if len(y_true) > 0 else 0.0
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
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
    entries = ordered_model_entries(results_root)
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

        csv_path = "metrics_summary.csv"
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        txt_path = "metrics_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for dname in sorted(out_df["dataset"].unique()):
                f.write(f"======== {dname} ========\n")
                sub = out_df[out_df["dataset"] == dname].copy()
                sub = sub.drop(columns=["dataset"])
                f.write(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
                f.write("\n\n")

        print("\n✅ 指标计算完成。")
        print(f"  - CSV: {csv_path}")
        print(f"  - TXT: {txt_path}")
    else:
        print("⚠️ 未生成任何指标结果，请检查文件路径与数据。")


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment results for all active models.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_sentiment(results_root=Path(args.results_root))
