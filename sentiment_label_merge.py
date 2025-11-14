# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline_utils import ordered_model_entries, resolve_dataset_base

DATASETS: List[Dict] = [
    {"name": "imdb", "file": "imdb_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/imdb"},
    {"name": "mental", "file": "mental_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/mental"},
    {"name": "news", "file": "news_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/news"},
    {"name": "fiqasa", "file": "fiqasa_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/fiqasa"},
    {"name": "imdb_sklearn", "file": "imdb_sklearn_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/imdb_sklearn"},
    {"name": "sst2", "file": "sst2_sentiment_results.csv", "pred_col": "prediction",
     "base_path": "results/sentiment/sst2"},
]

CANDIDATE_KEYS = ["text", "sentence", "review", "headline", "content", "input"]


def pick_merge_key(df: pd.DataFrame) -> str:
    for key in CANDIDATE_KEYS:
        if key in df.columns:
            return key
    raise KeyError(f"找不到合并键（未发现任一列：{CANDIDATE_KEYS}）")


def norm_text(value) -> str:
    value = "" if pd.isna(value) else str(value)
    return " ".join(value.strip().split())


def load_label_file(main_path: Path) -> Path | None:
    cand1 = main_path.with_suffix(".invalid.labeled.csv")
    cand2 = main_path.with_suffix(".invalid.labeled.test.csv")
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def pick_label_column(df_label: pd.DataFrame) -> str:
    if "normalized_label" in df_label.columns:
        return "normalized_label"
    if "sentiment_label" in df_label.columns:
        return "sentiment_label"
    if "label" in df_label.columns:
        return "label"
    raise KeyError("标注文件中未找到 normalized_label / sentiment_label / label 列")


def merge_corrected_labels(results_root: Path | str = "results"):
    results_root = Path(results_root)
    entries = ordered_model_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")
    model_dirs = [entry["display_name"] for entry in entries]

    for ds in DATASETS:
        base_dir = resolve_dataset_base(results_root, ds["base_path"])

        for model_dir in model_dirs:
            main_path = base_dir / model_dir / ds["file"]
            if not main_path.exists():
                print(f"⚠️ 缺少主文件：{main_path}")
                continue

            relabeled_marker = main_path.with_suffix(".relabeled.csv")
            if relabeled_marker.exists():
                print(f"⏭️ 已有 relabeled 文件，跳过：{relabeled_marker}")
                continue

            label_path = load_label_file(main_path)
            if not label_path:
                print(f"ℹ️ 无更正文件，跳过：{main_path}")
                continue

            df_main = pd.read_csv(main_path, dtype=str).fillna("")
            df_label = pd.read_csv(label_path, dtype=str).fillna("")

            try:
                key_main = pick_merge_key(df_main)
            except KeyError as exc:
                print(f"⚠️ {main_path} - {exc}")
                continue

            if key_main in df_label.columns:
                key_label = key_main
            else:
                try:
                    key_label = pick_merge_key(df_label)
                except KeyError:
                    print(f"⚠️ {label_path} - 找不到可用合并键")
                    continue

            label_col = pick_label_column(df_label)
            pred_col = ds["pred_col"]

            df_label["_k"] = df_label[key_label].map(norm_text)
            df_label = df_label[df_label["_k"] != ""]
            label_map = dict(zip(df_label["_k"], df_label[label_col]))

            df_main["_k"] = df_main[key_main].map(norm_text)
            repl = df_main["_k"].map(label_map)
            df_main[pred_col] = repl.fillna(df_main[pred_col])

            out_path = main_path.with_suffix(".relabeled.csv")
            df_main.drop(columns=["_k"], errors="ignore").to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ 保存：{out_path}")


if __name__ == "__main__":
    merge_corrected_labels()
