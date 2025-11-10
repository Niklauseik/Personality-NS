# -*- coding: utf-8 -*-
import os
import pandas as pd

models = {
    "base": "原始基座模型",
    "s":    "S性格模型",
    "n":    "N性格模型",
}

# ===== 全部数据集 =====
datasets = [
    {"name":"imdb",         "file":"imdb_sentiment_results.csv",        "pred_col":"prediction", "base_path":"results/sentiment/imdb"},
    {"name":"mental",       "file":"mental_sentiment_results.csv",      "pred_col":"prediction", "base_path":"results/sentiment/mental"},
    {"name":"news",         "file":"news_sentiment_results.csv",        "pred_col":"prediction", "base_path":"results/sentiment/news"},
    {"name":"fiqasa",       "file":"fiqasa_sentiment_results.csv",      "pred_col":"prediction", "base_path":"results/sentiment/fiqasa"},
    {"name":"imdb_sklearn", "file":"imdb_sklearn_sentiment_results.csv","pred_col":"prediction", "base_path":"results/sentiment/imdb_sklearn"},
    {"name":"sst2",         "file":"sst2_sentiment_results.csv",        "pred_col":"prediction", "base_path":"results/sentiment/sst2"},
]

# 优先使用 text 作为键；没有则依次尝试这些列名
CANDIDATE_KEYS = ["text", "sentence", "review", "headline", "content", "input"]

def pick_merge_key(df: pd.DataFrame) -> str:
    for k in CANDIDATE_KEYS:
        if k in df.columns:
            return k
    raise KeyError(f"找不到合并键（未发现任一列：{CANDIDATE_KEYS}）")

def norm(s):
    s = "" if pd.isna(s) else str(s)
    # 统一空白
    return " ".join(s.strip().split())

def load_label_file(main_path: str) -> str | None:
    """
    优先使用 *.invalid.labeled.csv；兼容旧后缀 *.invalid.labeled.test.csv
    """
    cand1 = main_path.replace(".csv", ".invalid.labeled.csv")
    cand2 = main_path.replace(".csv", ".invalid.labeled.test.csv")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    return None

def pick_label_column(df_label: pd.DataFrame) -> str:
    """
    兼容新旧命名：normalized_label（新）或 sentiment_label（旧）
    """
    if "normalized_label" in df_label.columns:
        return "normalized_label"
    if "sentiment_label" in df_label.columns:
        return "sentiment_label"
    # 兜底：若恰好有 label 列也可用
    if "label" in df_label.columns:
        return "label"
    raise KeyError("标注文件中未找到 normalized_label / sentiment_label / label 列")

# ========= 主流程 =========
for ds in datasets:
    for _, mfolder in models.items():
        main_path  = os.path.join(ds["base_path"], mfolder, ds["file"])
        if not os.path.exists(main_path):
            print(f"⚠️ 缺少主文件：{main_path}")
            continue

        label_path = load_label_file(main_path)
        if not label_path:
            print(f"ℹ️ 无更正文件，跳过：{main_path}")
            continue

        # 读主文件 & 更正文件
        df_main  = pd.read_csv(main_path, dtype=str).fillna("")
        df_label = pd.read_csv(label_path, dtype=str).fillna("")

        try:
            key_main = pick_merge_key(df_main)
        except KeyError as e:
            print(f"❌ {main_path} - {e}")
            continue

        try:
            key_label = pick_merge_key(df_label)
        except KeyError:
            # 如果标注文件用的列名和主文件一致，这里就沿用主文件的 key
            if key_main in df_label.columns:
                key_label = key_main
            else:
                print(f"❌ {label_path} - 找不到可用合并键")
                continue

        label_col = pick_label_column(df_label)
        pred_col  = ds["pred_col"]

        # 建立“文本 -> 新标签”的映射
        df_label["_k"] = df_label[key_label].map(norm)
        df_label = df_label[df_label["_k"] != ""]
        # 同一文本多条，更正以第一条为准（保持与之前一致）
        label_map = dict(zip(df_label["_k"], df_label[label_col]))

        # 将主文件的 prediction 按映射替换
        df_main["_k"] = df_main[key_main].map(norm)
        repl = df_main["_k"].map(label_map)
        df_main[pred_col] = repl.fillna(df_main[pred_col])

        # 输出
        out_path = main_path.replace(".csv", ".relabeled.csv")
        df_main.drop(columns=["_k"], errors="ignore").to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 保存：{out_path}")
