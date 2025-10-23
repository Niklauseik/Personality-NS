import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ========= 数据集配置（文件名不变，根目录会切换） =========
datasets = [
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["normal", "depression"],
        "label_col":  "label",
        "pred_col":   "prediction",
        "subdir":     "mental",
    },
    {
        "name":       "news_sentiment",
        "file":       "news_sentiment_results.csv",
        "label_map":  {"0": "bearish", "1": "bullish", "2": "neutral"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "subdir":     "news",
    },
    {
        "name":       "fiqasa_sentiment",
        "file":       "fiqasa_sentiment_results.csv",
        "label_map":  None,
        "allowed_labels": ["negative", "positive", "neutral"],
        "label_col":  "answer",
        "pred_col":   "prediction",
        "subdir":     "fiqasa",
    },
    {
        "name":       "imdb_sklearn",
        "file":       "imdb_sklearn_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "subdir":     "imdb_sklearn",
    },
    {
        "name":       "sst2",
        "file":       "sst2_sentiment_results.csv",
        "label_map":  {"0": "negative", "1": "positive"},
        "allowed_labels": None,
        "label_col":  "label",
        "pred_col":   "prediction",
        "subdir":     "sst2",
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
    earliest = None
    earliest_pos = 10**12
    for lbl in allowed:
        m = re.search(rf"\b{re.escape(lbl)}\b", text_l)
        if m and m.start() < earliest_pos:
            earliest = lbl
            earliest_pos = m.start()
    return earliest if earliest is not None else "invalid"

# ========= 主流程：对 sentiment 和 sentiment_pre 分别计算 =========
root_dirs = [
    ("results/sentiment", ""),         # 第一次实验，无后缀
    ("3B_results/sentiment", "_3B"), # 第二次实验，带 _pre 后缀
]

for root_dir, suffix in root_dirs:
    rows_cls = []

    print(f"\n🚀 开始处理目录：{root_dir}")

    for ds in datasets:
        print(f"🔍 数据集：{ds['name']}")
        allowed = build_allowed(ds)
        if not allowed:
            print(f"  ⚠️ {ds['name']} 未能解析到合法标签集合，跳过。")
            continue

        for mkey, mfolder in models.items():
            file_base = os.path.join(root_dir, ds["subdir"], mfolder, ds["file"])
            file_proc = file_base.replace(".csv", ".processed.csv")
            path = file_proc if os.path.exists(file_proc) else file_base

            if not os.path.exists(path):
                print(f"  ⚠️ 缺少文件：{path}")
                continue

            df = pd.read_csv(path)

            # 真实标签标准化 & 过滤
            y_true_all = map_true_label_series(ds, df[ds["label_col"]])
            mask_keep = y_true_all.isin(allowed)
            kept = df[mask_keep].copy()
            if kept.empty:
                print(f"  ⚠️ {ds['name']} - {mkey}: 无可评估样本。")
                continue

            kept["__pred_raw"] = kept[ds["pred_col"]].astype(str)
            kept["__pred_label"] = kept["__pred_raw"].apply(lambda x: extract_pred_label(x, allowed))

            y_true = map_true_label_series(ds, kept[ds["label_col"]]).tolist()
            y_pred = kept["__pred_label"].tolist()

            cm = confusion_matrix(y_true, y_pred, labels=allowed)
            tp = np.diag(cm).astype(float)
            support_true = cm.sum(axis=1).astype(float)
            pred_count   = cm.sum(axis=0).astype(float)

            precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count>0)
            recall    = np.divide(tp, support_true, out=np.zeros_like(tp), where=support_true>0)
            f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall)>0)

            for i, cls in enumerate(allowed):
                rows_cls.append({
                    "dataset": ds["name"],
                    "model": mkey,
                    "class": cls,
                    "support_true": int(support_true[i]),
                    "pred_count":   int(pred_count[i]),
                    "tp":           int(tp[i]),
                    "precision":    float(precision[i]),
                    "recall":       float(recall[i]),
                    "f1":           float(f1[i]),
                })

    # ========= 导出 =========
    if rows_cls:
        out_cls = pd.DataFrame(rows_cls)
        col_order = ["dataset", "model", "class",
                     "support_true", "pred_count", "tp",
                     "precision", "recall", "f1"]
        out_cls = out_cls[col_order]
        out_cls[["precision", "recall", "f1"]] = out_cls[["precision", "recall", "f1"]].round(2)

        csv_path = f"metrics_by_class{suffix}.csv"
        txt_path = f"metrics_by_class{suffix}.txt"

        out_cls.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(txt_path, "w", encoding="utf-8") as f:
            for dname in sorted(out_cls["dataset"].unique()):
                f.write(f"======== {dname} ========\n")
                sub = out_cls[out_cls["dataset"] == dname].copy()
                name_map = {"base": "基座模型", "f": "F模型", "t": "T模型"}
                sub["model"] = sub["model"].map(name_map).fillna(sub["model"])
                sub = sub.sort_values(["model", "class"])
                f.write(sub.to_string(index=False))
                f.write("\n\n")

        print(f"✅ 已完成 {root_dir}")
        print(f"  - CSV: {csv_path}")
        print(f"  - TXT: {txt_path}")
    else:
        print(f"⚠️ {root_dir} 未生成任何结果。")
