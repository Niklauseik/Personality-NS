# -*- coding: utf-8 -*-
import os
import re
from collections import Counter, defaultdict
import pandas as pd

# ========= 数据集配置 =========
datasets = [
    {"name":"imdb_sentiment","file":"imdb_sentiment_results.csv",
     "label_map":{"0":"negative","1":"positive"},
     "allowed_labels":None,"label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/imdb"},
    {"name":"mental_sentiment","file":"mental_sentiment_results.csv",
     "label_map":None,"allowed_labels":["normal","depression"],
     "label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/mental"},
    {"name":"news_sentiment","file":"news_sentiment_results.csv",
     "label_map":{"0":"bearish","1":"bullish","2":"neutral"},
     "allowed_labels":None,"label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/news"},
    {"name":"fiqasa_sentiment","file":"fiqasa_sentiment_results.csv",
     "label_map":None,"allowed_labels":["negative","positive","neutral"],
     "label_col":"answer","pred_col":"prediction",
     "base_path":"results/sentiment/fiqasa"},
    {"name":"imdb_sklearn","file":"imdb_sklearn_sentiment_results.csv",
     "label_map":{"0":"negative","1":"positive"},
     "allowed_labels":None,"label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/imdb_sklearn"},
    {"name":"sst2","file":"sst2_sentiment_results.csv",
     "label_map":{"0":"negative","1":"positive"},
     "allowed_labels":None,"label_col":"label","pred_col":"prediction",
     "base_path":"results/sentiment/sst2"},
]

# ========= 模型文件夹名 =========
models = {
    "base": "原始基座模型",
    "s": "S性格模型",
    "n": "N性格模型",
}

# 仅去掉“末尾”句号(. / 。 / …)，并小写；不移除其它标点
TAILING_DOTS = re.compile(r"[\.。…]+$")

def norm_token(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = TAILING_DOTS.sub("", s)
    return s.lower()

def determine_true_labels_from_df(ds_cfg: dict, df_true: pd.DataFrame) -> list[str]:
    if ds_cfg["allowed_labels"]:
        return sorted({norm_token(x) for x in ds_cfg["allowed_labels"]})
    if ds_cfg["label_map"]:
        return sorted({norm_token(v) for v in ds_cfg["label_map"].values()})
    return sorted({norm_token(x) for x in df_true[ds_cfg["label_col"]].astype(str)})

def pick_pred_path(base_path: str) -> str | None:
    """预测文件优先级：.relabeled.csv > .processed.csv > .csv"""
    relabeled = base_path.replace(".csv", ".relabeled.csv")
    processed = base_path.replace(".csv", ".processed.csv")
    for p in (relabeled, processed, base_path):
        if os.path.exists(p):
            return p
    return None

def classify_prediction_strict(pred_text: str, valid_pred_set: set[str]) -> str:
    t = norm_token(pred_text)
    return t if t in valid_pred_set else "invalid"

# ===== 统计容器 =====
dist_all = defaultdict(lambda: defaultdict(lambda: {"true": 0, "base": 0, "s": 0, "n": 0}))
label_order_map: dict[str, list[str]] = {}

for ds in datasets:
    print(f"[INFO] Processing dataset: {ds['name']}")

    # ---- 1) 始终从“原始源文件”读取真实标签与真实计数 ----
    base_src = os.path.join(ds["base_path"], "原始基座模型", ds["file"])  # 用任意模型文件都可；最好选存在率高的，如原始基座模型
    # 若该模型缺失，用其它模型的原始文件兜底
    if not os.path.exists(base_src):
        for _, mfolder_try in models.items():
            cand = os.path.join(ds["base_path"], mfolder_try, ds["file"])
            if os.path.exists(cand):
                base_src = cand
                break
    if not os.path.exists(base_src):
        print(f"  [WARN] Missing source file for labels: {base_src}")
        continue

    df_true = pd.read_csv(base_src, dtype=str).fillna("")
    if ds["label_map"]:
        df_true[ds["label_col"]] = df_true[ds["label_col"]].astype(str).map(ds["label_map"]).fillna("")
    df_true[ds["label_col"]] = df_true[ds["label_col"]].astype(str).map(norm_token)

    true_labels = determine_true_labels_from_df(ds, df_true)
    # 预测允许集合 = 真实标签 ∪ {neutral, mixed, invalid}（有的集没有 invalid 也没关系，只在预测侧出现）
    ordered = true_labels + [x for x in ["neutral", "mixed", "invalid"] if x not in true_labels]
    valid_pred_set = set(ordered)

    label_order_map[ds["name"]] = ordered
    for lbl in ordered:
        _ = dist_all[ds["name"]][lbl]

    # 真实计数：严格只对 true_labels 统计
    counts_true = Counter(df_true[ds["label_col"]])
    for lbl in true_labels:
        dist_all[ds["name"]][lbl]["true"] = int(counts_true.get(lbl, 0))

    # ---- 2) 遍历每个模型，读取预测文件（优先 relabeled）并统计预测分布 ----
    for mkey, mfolder in models.items():
        raw_base = os.path.join(ds["base_path"], mfolder, ds["file"])
        pred_path = pick_pred_path(raw_base)
        if not pred_path:
            print(f"  [WARN] Missing all variants for predictions: {raw_base}")
            continue

        df_pred = pd.read_csv(pred_path, dtype=str).fillna("")

        # label_map 仅影响真实标签，不改预测列
        for pred in df_pred[ds["pred_col"]].astype(str):
            category = classify_prediction_strict(pred, valid_pred_set)
            dist_all[ds["name"]][category][mkey] += 1

# ===== 输出 =====
outfile = "label_distribution_summary.txt"
with open(outfile, "w", encoding="utf-8") as f:
    for dname, label_dict in dist_all.items():
        if dname not in label_order_map:
            continue
        f.write(f"======== {dname} ========\n")
        df_out = (
            pd.DataFrame(label_dict).T
            .fillna(0).astype(int)
            .loc[label_order_map[dname], ["true", "base", "s", "n"]]
            .rename(columns={"true":"真实数量","base":"基座模型","s":"S模型","n":"N模型"})
        )
        f.write(df_out.to_string())
        f.write("\n\n")

print(f"\n[INFO] Summary saved to {outfile}")
