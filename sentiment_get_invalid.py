# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

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

TAILING_DOTS_PATTERN = re.compile(r"[\.。…]+$")


def norm_label(value: str) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    cleaned = TAILING_DOTS_PATTERN.sub("", cleaned)
    return cleaned.lower()


def _derive_allowed_labels(df: pd.DataFrame, cfg: Dict, base_allowed: Optional[Set[str]]) -> Set[str]:
    if base_allowed:
        return base_allowed
    if cfg["label_map"] is not None:
        return {norm_label(v) for v in cfg["label_map"].values()}
    return set(df[cfg["label_col"]].map(norm_label).unique())


def collect_invalid_predictions(results_root: Path | str = "results") -> None:
    results_root = Path(results_root)
    entries = ordered_model_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")

    model_folders = [entry["display_name"] for entry in entries]

    for cfg in DATASETS:
        print(f"🔍 处理数据集：{cfg['name']}")
        allowed = set() if cfg["allowed_labels"] is None else {norm_label(x) for x in cfg["allowed_labels"]}
        base_dir = resolve_dataset_base(results_root, cfg["base_path"])

        for model_folder in model_folders:
            path = base_dir / model_folder / cfg["file"]
            if not path.exists():
                print(f"  ⚠️ 缺少文件：{path}")
                continue

            invalid_path = path.with_suffix(".invalid.csv")
            labeled_path = path.with_suffix(".invalid.labeled.csv")
            relabeled_path = path.with_suffix(".relabeled.csv")

            if invalid_path.exists() or labeled_path.exists() or relabeled_path.exists():
                print(f"  ⏭️ 已存在 invalid/labeled/relabeled 文件，跳过：{model_folder}")
                continue

            df = pd.read_csv(path, dtype=str).fillna("")
            original_df = df.copy()

            if cfg["label_map"] is not None:
                df[cfg["label_col"]] = df[cfg["label_col"]].astype(str).map(cfg["label_map"])

            df[cfg["label_col"]] = df[cfg["label_col"]].map(norm_label)
            df["cleaned_pred"] = df[cfg["pred_col"]].map(norm_label)

            allowed = _derive_allowed_labels(df, cfg, allowed if allowed else None)
            extra_ok = {"mixed", "neutral"}
            is_valid = df["cleaned_pred"].isin(allowed.union(extra_ok))

            invalid_df = original_df[~is_valid].copy()
            if not invalid_df.empty:
                invalid_df.to_csv(invalid_path, index=False)
                print(f"  🚫 非法 prediction 条目已保存至：{invalid_path}")
            else:
                print("  ✅ 所有 prediction 都是规范标签（含 mixed/neutral）")


if __name__ == "__main__":
    collect_invalid_predictions()
