# -*- coding: utf-8 -*-
import os
import re
import time
import pandas as pd
from typing import List, Dict, Set
from openai import OpenAI

# ===== OpenAI 配置 =====
OPENAI_API_KEY = ""  # ← 填你的 Key
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 数据集配置（与你之前一致）=====
DATASETS: List[Dict] = [
    {"name": "imdb", "file": "imdb_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": os.path.join("results", "sentiment", "imdb")},
    {"name": "mental", "file": "mental_sentiment_results.csv",
     "allowed": ["normal", "depression"],
     "base_path": os.path.join("results", "sentiment", "mental")},
    {"name": "news", "file": "news_sentiment_results.csv",
     "allowed": ["bearish", "bullish", "neutral"],
     "base_path": os.path.join("results", "sentiment", "news")},
    {"name": "fiqasa", "file": "fiqasa_sentiment_results.csv",
     "allowed": ["negative", "positive", "neutral"],
     "base_path": os.path.join("results", "sentiment", "fiqasa")},
    {"name": "imdb_sklearn", "file": "imdb_sklearn_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": os.path.join("results", "sentiment", "imdb_sklearn")},
    {"name": "sst2", "file": "sst2_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": os.path.join("results", "sentiment", "sst2")},
]

# 仅去掉“末尾”句号（英文. / 中文。 / 省略号…），忽略大小写；不移除其它标点
TAILING_DOTS_PATTERN = re.compile(r'[\.。…]+$')

def norm_token(s: str) -> str:
    s = str(s).strip()
    s = TAILING_DOTS_PATTERN.sub("", s)
    return s.lower()

def build_choice_set(allowed: List[str]) -> Set[str]:
    # 允许集合 = allowed ∪ {mixed, neutral, invalid}
    base = {norm_token(x) for x in allowed}
    base.update({"mixed", "neutral", "invalid"})
    return base

def build_context_prompt(dataset_name: str) -> str:
    if dataset_name == "imdb":
        return ("You are a movie review sentiment classifier. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    elif dataset_name == "sst2":
        return ("You are a sentence-level sentiment classifier. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    elif dataset_name == "imdb_sklearn":
        return ("You classify user-written movie reviews. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    elif dataset_name == "fiqasa":
        return ("You are a financial sentiment classifier. "
                "Valid labels: positive, neutral, negative. Also allow: mixed, neutral, invalid.")
    elif dataset_name == "news":
        return ("You analyze financial news headlines. "
                "Valid labels: bearish, bullish, neutral. Also allow: mixed, neutral, invalid.")
    elif dataset_name == "mental":
        # ★ 特别注明：出现 suicidal / suicide / lifeline 则必为 depression
        return ("You classify short social posts about mental health. "
                "Valid labels: normal, depression. Also allow: mixed, neutral, invalid. "
                "HARD RULE: If the text contains 'suicidal', 'suicide', or 'lifeline', "
                "classify it as 'depression'.")
    else:
        return ("Classify the text into one of the valid labels of the dataset; "
                "Also allow: mixed, neutral, invalid.")


def classify_with_gpt(dataset_name: str, pred_text: str, choices: Set[str]) -> str:
    """
    让 GPT 在 choices 中挑一个：allowed ∪ {mixed, neutral, invalid}
    强约束返回：只返回 choices 里的一个词。
    """
    context = build_context_prompt(dataset_name)
    choices_list = sorted(list(choices))
    choices_str = ", ".join(choices_list)

    user_prompt = (
        f"{context}\n\n"
        f"Your task: Map the following model output to EXACTLY ONE of these labels:\n"
        f"{choices_str}\n\n"
        f"Rules:\n"
        f"1) If it clearly expresses a valid sentiment, choose that sentiment.\n"
        f"2) If it mixes multiple sentiments or hedges between them, choose 'mixed'.\n"
        f"3) If it is unclear, evasive, meta-commentary, or lacks necessary info, choose 'invalid'.\n"
        f"4) Respond with ONE word only, exactly as in the list.\n\n"
        f"Model output:\n\"\"\"\n{pred_text}\n\"\"\"\n\n"
        f"Answer with ONE WORD ONLY:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system",
                 "content": f"Respond with exactly ONE token from this set: {choices_str}. No explanation."},
                {"role": "user", "content": user_prompt},
            ],
        )
        label = norm_token(resp.choices[0].message.content)
        return label if label in choices else "invalid"
    except Exception as e:
        print(f"❌ GPT error: {e}")
        return "invalid"

def direct_or_gpt(dataset_name: str, pred_text: str, choices: Set[str]) -> str:
    """
    先做直接匹配（大小写不敏感、仅去尾句号）；失败才走 GPT。
    """
    token = norm_token(pred_text)
    if token in choices:
        return token
    return classify_with_gpt(dataset_name, pred_text, choices)

def process_all():
    model_dirs = ["S性格模型", "N性格模型", "原始基座模型"]

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_file = ds["file"]
        allowed = ds["allowed"]
        choices = build_choice_set(allowed)

        print(f"\n============================")
        print(f"📚 Dataset: {ds_name}")
        print(f"📝 Allowed: {allowed} | Extra allowed: ['mixed','neutral','invalid']")

        for model_dir in model_dirs:
            folder = os.path.join(ds["base_path"], model_dir)
            input_path = os.path.join(folder, ds_file.replace(".csv", ".invalid.csv"))
            output_path = os.path.join(folder, ds_file.replace(".csv", ".invalid.labeled.csv"))

            if not os.path.exists(input_path):
                print(f"  ⚠️ Missing: {input_path}")
                continue

            print(f"  📄 Processing invalids -> {model_dir}")
            df = pd.read_csv(input_path)
            if "prediction" not in df.columns:
                print(f"  ❌ 'prediction' column not found in {input_path}")
                continue

            labels: List[str] = []
            for i, row in df.iterrows():
                pred = str(row.get("prediction", "")).strip()
                label = direct_or_gpt(ds_name, pred, choices)
                labels.append(label)
                head = pred.replace("\n", " ")[:80]
                print(f"   [{i+1:>5}] {head} -> {label}")
                time.sleep(0.25)  # 轻微限速，避免触发限流

            df["normalized_label"] = labels
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ Saved: {output_path}")

if __name__ == "__main__":
    process_all()
