import os
import re
import pandas as pd
from collections import Counter, defaultdict

# ========= 数据集配置（保持你的原样） =========
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
    {"name":"fiqasa_sentiment","file":"fiqasa_fiqasa_sentiment_results.csv".replace("_fiqasa",""),
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

models = {"base":"原始基座模型","n":"N性格模型","s":"S性格模型"}

# ====== 只做“大小写无关 + 去掉两端空白/句号等标点”的规范化 ======
TRIM_PUNCT = r'^[\s\.\,\;\:\!\?\-\'"“”‘’\(\)\[\]\{\}]+|[\s\.\,\;\:\!\?\-\'"“”‘’\(\)\[\]\{\}]+$'

def norm_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(TRIM_PUNCT, "", s.strip())
    return s.lower()

dist_all = defaultdict(lambda: defaultdict(lambda: {"true":0, "base":0, "f":0, "t":0}))

for ds in datasets:
    print(f"🔍 处理数据集：{ds['name']}")
    allowed = set() if ds["allowed_labels"] is None else {norm_label(x) for x in ds["allowed_labels"]}
    true_done = False

    for mkey, mfolder in models.items():
        path = os.path.join(ds["base_path"], mfolder, ds["file"])
        if not os.path.exists(path):
            print(f"  ⚠️ 缺少文件：{path}")
            continue

        df = pd.read_csv(path, dtype=str).fillna("")
        original_df = df.copy()

        # 真实标签映射（如有）
        if ds["label_map"] is not None:
            df[ds["label_col"]] = df[ds["label_col"]].astype(str).map(ds["label_map"])

        # 规范化：真实标签 & 预测，仅用于比较
        df[ds["label_col"]] = df[ds["label_col"]].map(norm_label)
        df["cleaned_pred"]  = df[ds["pred_col"]].map(norm_label)

        # 若没提供 allowed，则用映射值或真实标签集合
        if not allowed:
            if ds["label_map"] is not None:
                allowed = {norm_label(v) for v in ds["label_map"].values()}
            else:
                allowed = set(df[ds["label_col"]].unique())

        # 统计真实分布
        if not true_done:
            for lbl, cnt in Counter(df[ds["label_col"]]).items():
                if lbl in allowed:
                    dist_all[ds["name"]][lbl]["true"] = cnt
            true_done = True

        # 预测分布 —— 【严格等值匹配】不再用子串包含
        for lbl in allowed:
            match_count = (df["cleaned_pred"] == lbl).sum()
            dist_all[ds["name"]][lbl][mkey] = match_count

        # 保存非法预测（完全不等于任一允许标签）
        is_valid = df["cleaned_pred"].isin(allowed)
        invalid_df = original_df[~is_valid].copy()
        invalid_path = os.path.join(ds["base_path"], mfolder, ds["file"].replace(".csv", ".invalid.csv"))
        if not invalid_df.empty:
            invalid_df.to_csv(invalid_path, index=False)
            print(f"  🚫 非法 prediction 条目已保存至：{invalid_path}")
        else:
            print(f"  ✅ 所有 prediction 都是规范标签")
