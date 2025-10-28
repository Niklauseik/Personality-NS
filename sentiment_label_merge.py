import os
import pandas as pd

models = {
    "base": "原始基座模型",
    "s":    "S性格模型",
    "n":    "N性格模型",
}

datasets = [
    {
        "name":       "mental_sentiment",
        "file":       "mental_sentiment_results.csv",
        "pred_col":   "prediction",
        "merge_key":  "text",
        "base_path":  "results/sentiment/mental",
        "label_suffix": ".invalid.labeled.test.csv",
    },
]

def norm(s):
    s = "" if pd.isna(s) else str(s)
    return " ".join(s.strip().split())

for ds in datasets:
    for _, mfolder in models.items():
        main_path  = os.path.join(ds["base_path"], mfolder, ds["file"])
        label_path = main_path.replace(".csv", ds["label_suffix"])
        if not (os.path.exists(main_path) and os.path.exists(label_path)):
            continue

        df_main  = pd.read_csv(main_path, dtype=str).fillna("")
        df_label = pd.read_csv(label_path, dtype=str).fillna("")

        # 建立 text -> sentiment_label 的映射（只用来替换）
        key = ds["merge_key"]
        pred_col = ds["pred_col"]

        df_label["_k"] = df_label[key].map(norm)
        df_label = df_label[df_label["_k"] != ""]
        # 若同一 text 多条，保留第一条
        label_map = dict(zip(df_label["_k"], df_label["sentiment_label"]))

        # 按映射替换 prediction；无映射则保持原值
        df_main["_k"] = df_main[key].map(norm)
        repl = df_main["_k"].map(label_map)
        df_main[pred_col] = repl.fillna(df_main[pred_col])

        # 输出新文件，结构不变
        out_path = main_path.replace(".csv", ".relabeled.csv")
        df_main.drop(columns=["_k"], errors="ignore").to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 保存：{out_path}")
