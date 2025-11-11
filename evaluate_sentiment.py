# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ========= 模型配置（一次定义，含目录+展示名）=========
MODELS = {
    "base": {"folder": "原始基座模型", "display": "基座模型"},
    "s":    {"folder": "S性格模型",   "display": "S模型"},
    "n":    {"folder": "N性格模型",   "display": "N模型"},
}

# ========= 数据集配置 =========
datasets = [
    {"name":"imdb_sentiment","file":"imdb_sentiment_results.csv",
     "label_map":{"0":"positive","1":"negative"},
     "allowed_labels":None, "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/imdb"},
    {"name":"mental_sentiment","file":"mental_sentiment_results.csv",
     "label_map":None, "allowed_labels":["normal","depression"],
     "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/mental"},
    {"name":"news_sentiment","file":"news_sentiment_results.csv",
     "label_map":{"0":"bearish","1":"bullish","2":"neutral"},
     "allowed_labels":None, "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/news"},
    {"name":"fiqasa_sentiment","file":"fiqasa_sentiment_results.csv",
     "label_map":None, "allowed_labels":["negative","positive","neutral"],
     "label_col":"answer","pred_col":"prediction",
     "base_path":"results/sentiment/fiqasa"},
    {"name":"imdb_sklearn","file":"imdb_sklearn_sentiment_results.csv",
     "label_map":{"0":"negative","1":"positive"},
     "allowed_labels":None, "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/imdb_sklearn"},
    {"name":"sst2","file":"sst2_sentiment_results.csv",
     "label_map":{"0":"negative","1":"positive"},
     "allowed_labels":None, "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/sst2"},
]

# ========= 工具函数 =========
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

def map_true_label_series(ds, s: pd.Series) -> pd.Series:
    if ds["label_map"]:
        s = s.astype(str).map(ds["label_map"])
    return s.astype(str).apply(clean)

def extract_pred_label(text: str, allowed: list) -> str:
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    text_l = text.lower()
    earliest, pos = None, 10**12
    for lbl in allowed:
        m = re.search(rf"\b{re.escape(lbl)}\b", text_l)
        if m and m.start() < pos:
            earliest, pos = lbl, m.start()
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

def pick_pred_path(base_path: str) -> str | None:
    """优先读取纠正合并后的文件；若不存在则读原始结果。"""
    relabeled = base_path.replace(".csv", ".relabeled.csv")
    if os.path.exists(relabeled):
        return relabeled
    return base_path if os.path.exists(base_path) else None

# ========= 主流程 =========
rows = []

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    allowed = build_allowed(ds)
    if not allowed:
        print(f"  ⚠️ 数据集 {ds['name']} 未能解析到合法标签集合，跳过。")
        continue

    for mkey, mconf in MODELS.items():
        mfolder = mconf["folder"]
        path = pick_pred_path(os.path.join(ds["base_path"], mfolder, ds["file"]))
        if not path:
            print(f"  ⚠️ 缺少文件：{os.path.join(ds['base_path'], mfolder, ds['file'])}")
            continue

        df = pd.read_csv(path)

        # 真实标签 -> 清洗并仅保留在 allowed 内的样本
        y_true_all = map_true_label_series(ds, df[ds["label_col"]])
        mask_keep = y_true_all.isin(allowed)
        kept = df[mask_keep].copy()
        if kept.empty:
            print(f"  ⚠️ {ds['name']} - {mkey}: 无可评估样本。")
            continue

        # 预测文本 -> 抽取到合法标签（若抽不到为 invalid，但不会计入 class_labels 的PRF）
        kept["__pred_raw"] = kept[ds["pred_col"]].astype(str)
        kept["__pred_label"] = kept["__pred_raw"].apply(lambda x: extract_pred_label(x, allowed))

        y_true = map_true_label_series(ds, kept[ds["label_col"]]).tolist()
        y_pred = kept["__pred_label"].tolist()

        metrics = compute_metrics(y_true, y_pred, class_labels=allowed)

        rows.append({
            "dataset": ds["name"],
            "model": mkey,
            "labels": "|".join(allowed),
            **metrics
        })

# ========= 导出汇总 =========
if rows:
    out_df = pd.DataFrame(rows)

    col_order = [
        "dataset", "model", "labels", "support",
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
            sub["model"] = sub["model"].map(lambda k: MODELS[k]["display"]).fillna(sub["model"])
            f.write(sub.drop(columns=["dataset"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            f.write("\n\n")

    print("\n✅ 指标计算完成！")
    print(f"  - CSV: {csv_path}")
    print(f"  - TXT: {txt_path}")
else:
    print("⚠️ 未生成任何指标结果，请检查文件路径与数据。")
