import os
import re
from collections import Counter, defaultdict

import pandas as pd

# ========= 数据集配置 =========
datasets = [
    {
        "name": "imdb_sentiment",
        "file": "imdb_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/imdb",
    },
    {
        "name": "mental_sentiment",
        "file": "mental_sentiment_results.csv",
        "label_map": None,
        "allowed_labels": ["normal", "depression"],
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/mental",
    },
    {
        "name": "news_sentiment",
        "file": "news_sentiment_results.csv",
        "label_map": {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/news",
    },
    {
        "name": "fiqasa_sentiment",
        "file": "fiqasa_sentiment_results.csv",
        "label_map": None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col": "answer",
        "pred_col": "prediction",
        "base_path": "results/sentiment/fiqasa",
    },
    {
        "name": "imdb_sklearn",
        "file": "imdb_sklearn_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/imdb_sklearn",
    },
    {
        "name": "sst2",
        "file": "sst2_sentiment_results.csv",
        "label_map": {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col": "label",
        "pred_col": "prediction",
        "base_path": "results/sentiment/sst2",
    },
]

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "f": "F性格模型",
    "t": "T性格模型",
}


def clean(text: str) -> str:
    """Return a lowercase string with only alphabetic characters."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())


def determine_true_labels(ds_cfg: dict, df: pd.DataFrame) -> list:
    if ds_cfg["allowed_labels"]:
        return sorted(clean(lbl) for lbl in ds_cfg["allowed_labels"])
    if ds_cfg["label_map"]:
        return sorted(clean(lbl) for lbl in ds_cfg["label_map"].values())
    return sorted(clean(lbl) for lbl in df[ds_cfg["label_col"]].unique())


def classify_prediction(pred_text: str, candidate_labels: set) -> str:
    clean_pred = clean(pred_text)
    hits = [lbl for lbl in candidate_labels if lbl in clean_pred]

    if "mixed" in hits:
        return "mixed"

    non_neutral_hits = [lbl for lbl in hits if lbl not in {"neutral", "mixed"}]
    has_neutral = "neutral" in hits

    if len(non_neutral_hits) > 1:
        return "mixed"
    if non_neutral_hits and has_neutral:
        return "mixed"
    if has_neutral and not non_neutral_hits:
        return "neutral"
    if non_neutral_hits:
        return non_neutral_hits[0]
    if "invalid" in clean_pred:
        return "invalid"
    return "invalid"


dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "f": 0, "t": 0}))
label_order_map = {}

for ds in datasets:
    print(f"[INFO] Processing dataset: {ds['name']}")
    true_labels = None
    candidate_labels = None
    true_count_logged = False

    for mkey, mfolder in models.items():
        file_base = os.path.join(ds["base_path"], mfolder, ds["file"])
        file_proc = file_base.replace(".csv", ".processed.csv")
        path = file_proc if os.path.exists(file_proc) else file_base

        if not os.path.exists(path):
            print(f"  [WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        if ds["label_map"]:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        df[ds["label_col"]] = df[ds["label_col"]].astype(str).apply(clean)

        if true_labels is None:
            true_labels = determine_true_labels(ds, df)
            candidate_labels = set(true_labels) | {"neutral", "mixed"}
            extras = [lbl for lbl in ("neutral", "mixed", "invalid") if lbl not in true_labels]
            label_order_map[ds["name"]] = true_labels + extras
            for lbl in label_order_map[ds["name"]]:
                _ = dist_all[ds["name"]][lbl]

        if true_labels is None or candidate_labels is None:
            continue

        if not true_count_logged:
            counts = Counter(df[ds["label_col"]])
            for lbl in true_labels:
                dist_all[ds["name"]][lbl]["true"] = int(counts.get(lbl, 0))
            true_count_logged = True

        for pred in df[ds["pred_col"]].astype(str):
            category = classify_prediction(pred, candidate_labels)
            dist_all[ds["name"]][category][mkey] += 1

outfile = "label_distribution_summary.txt"
with open(outfile, "w", encoding="utf-8") as f:
    for dname, label_dict in dist_all.items():
        if dname not in label_order_map:
            continue
        f.write(f"======== {dname} ========\n")
        df_out = (
            pd.DataFrame(label_dict).T
            .fillna(0)
            .astype(int)
            .loc[label_order_map[dname], ["true", "base", "f", "t"]]
            .rename(columns={
                "true": "真实数量",
                "base": "基座模型",
                "f": "F模型",
                "t": "T模型",
            })
        )
        f.write(df_out.to_string())
        f.write("\n\n")

print(f"\n[INFO] Summary saved to {outfile}")
