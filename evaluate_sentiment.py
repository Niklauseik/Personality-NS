import os
import re
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ========= 数据集配置 =========
datasets = [
    {
        "name":       "imdb_sentiment",
        "file":       "imdb_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,  # -> 使用 label_map 的值作为合法标签
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb",
    },
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["normal", "depression"],
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/mental",
    },
    {
        "name":       "news_sentiment",
        "file":       "news_sentiment_results.csv",
        "label_map":  {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,  # -> 使用 label_map 的值作为合法标签
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/news",
    },
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col":  "answer",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/fiqasa",
    },
    {
        "name":       "imdb_sklearn",
        "file":       "imdb_sklearn_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,  # -> 使用 label_map 的值作为合法标签
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/imdb_sklearn",
    },
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,  # -> 使用 label_map 的值作为合法标签
        "label_col":  "label",
        "pred_col":   "prediction",
        "base_path":  "results/sentiment/sst2",
    },
]

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "f":    "F性格模型",
    "t":    "T性格模型",
}

# ========= 工具函数 =========
def clean(text: str) -> str:
    """只保留字母并小写，用于标准化标签"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())

def build_allowed(ds) -> list:
    """得到数据集的合法标签（清洗后），有 allowed_labels 优先；否则来自 label_map 的值"""
    if ds["allowed_labels"]:
        allowed = [clean(x) for x in ds["allowed_labels"]]
    elif ds["label_map"]:
        allowed = [clean(x) for x in ds["label_map"].values()]
    else:
        allowed = []
    return sorted(set(allowed))

def map_true_label_series(ds, s: pd.Series) -> pd.Series:
    """将真实标签列映射并清洗"""
    if ds["label_map"]:
        s = s.astype(str).map(ds["label_map"])  # 数字字符串 -> 文字
    return s.astype(str).apply(clean)

def extract_pred_label(text: str, allowed: list) -> str:
    """
    从自由文本 prediction 中抽取第一个出现的合法标签（整词匹配）。
    若无匹配，返回 'invalid'。
    若出现多个，取最早出现的那个。
    """
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    text_l = text.lower()
    earliest = None
    earliest_pos = 10**12
    for lbl in allowed:
        m = re.search(rf"\b{re.escape(lbl)}\b", text_l)
        if m and m.start() < earliest_pos:
            earliest = lbl
            earliest_pos = m.start()
    return earliest if earliest is not None else "invalid"

def compute_metrics(y_true, y_pred, class_labels):
    """
    返回 accuracy、precision/recall/f1（macro/weighted）
    注意：macro/weighted 仅在 class_labels 上计算，不含 'invalid'
    """
    import numpy as np
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if len(y_true) > 0 else 0.0

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weight),
        "recall_weighted": float(r_weight),
        "f1_weighted": float(f_weight),
        "support": int(len(y_true)),
    }

# ========= 主流程 =========
rows = []

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    allowed = build_allowed(ds)
    if not allowed:
        print(f"  ⚠️ 数据集 {ds['name']} 未能解析到合法标签集合，跳过。")
        continue

    for mkey, mfolder in models.items():
        file_base = os.path.join(ds["base_path"], mfolder, ds["file"])
        file_proc = file_base.replace(".csv", ".processed.csv")
        path = file_proc if os.path.exists(file_proc) else file_base

        if not os.path.exists(path):
            print(f"  ⚠️ 缺少文件：{path}")
            continue

        df = pd.read_csv(path)

        # 真实标签标准化 & 过滤到合法集合
        y_true_all = map_true_label_series(ds, df[ds["label_col"]])
        mask_keep = y_true_all.isin(allowed)
        kept = df[mask_keep].copy()
        if kept.empty:
            print(f"  ⚠️ {ds['name']} - {mkey}: 无可评估样本（真实标签不在合法集合）。")
            continue

        # 预测抽取 -> y_pred
        kept["__pred_raw"] = kept[ds["pred_col"]].astype(str)
        kept["__pred_label"] = kept["__pred_raw"].apply(lambda x: extract_pred_label(x, allowed))

        # 得到 y_true / y_pred
        y_true = map_true_label_series(ds, kept[ds["label_col"]]).tolist()
        y_pred = kept["__pred_label"].tolist()

        # 计算指标
        metrics = compute_metrics(y_true, y_pred, class_labels=allowed)

        rows.append({
            "dataset": ds["name"],
            "model": mkey,  # base/f/t
            "labels": "|".join(allowed),
            **metrics
        })

# ========= 导出汇总（两位小数） =========
if rows:
    out_df = pd.DataFrame(rows)

    # 统一列顺序
    col_order = [
        "dataset", "model", "labels", "support",
        "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    out_df = out_df[col_order]

    # 四舍五入保留两位小数
    num_cols = [
        "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    out_df[num_cols] = out_df[num_cols].round(2)

    # 保存 CSV
    csv_path = "metrics_summary.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存 TXT（强制两位小数显示）
    txt_path = "metrics_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for dname in sorted(out_df["dataset"].unique()):
            f.write(f"======== {dname} ========\n")
            sub = out_df[out_df["dataset"] == dname].copy()
            name_map = {"base": "基座模型", "f": "F模型", "t": "T模型"}
            sub["model"] = sub["model"].map(name_map).fillna(sub["model"])
            f.write(sub.drop(columns=["dataset"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            f.write("\n\n")

    print("\n✅ 指标计算完成！")
    print(f"  - CSV: {csv_path}")
    print(f"  - TXT: {txt_path}")
else:
    print("⚠️ 未生成任何指标结果，请检查文件路径与数据。")
