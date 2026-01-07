# -*- coding: utf-8 -*-
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from openai import OpenAI

from pipeline_utils import ordered_sentiment_entries, resolve_dataset_base

# ===== OpenAI 配置 =====
ENV_PATH = Path(__file__).with_name(".env")

def load_openai_api_key(env_path: Path) -> str:
    """Load API key from .env file first, fallback to OPENAI_API_KEY env var."""
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if key.strip() != "OPENAI_API_KEY" or not sep:
                continue
            clean_value = value.split("#", 1)[0].strip().strip("\"'")
            if clean_value:
                return clean_value

    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value

    raise RuntimeError(
        f"Missing OpenAI API Key. Set OPENAI_API_KEY in {env_path} or export the OPENAI_API_KEY env var."
    )

OPENAI_API_KEY = load_openai_api_key(ENV_PATH)

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 数据集配置 =====
DATASETS: List[Dict] = [
    {"name": "imdb", "file": "imdb_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": "results/sentiment/imdb"},
    {"name": "mental", "file": "mental_sentiment_results.csv",
     "allowed": ["normal", "depression"],
     "base_path": "results/sentiment/mental"},
    {"name": "news", "file": "news_sentiment_results.csv",
     "allowed": ["bearish", "bullish", "neutral"],
     "base_path": "results/sentiment/news"},
    {"name": "fiqasa", "file": "fiqasa_sentiment_results.csv",
     "allowed": ["negative", "positive", "neutral"],
     "base_path": "results/sentiment/fiqasa"},
    {"name": "imdb_sklearn", "file": "imdb_sklearn_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": "results/sentiment/imdb_sklearn"},
    {"name": "sst2", "file": "sst2_sentiment_results.csv",
     "allowed": ["positive", "negative"],
     "base_path": "results/sentiment/sst2"},
]

TAILING_DOTS_PATTERN = re.compile(r"[\.。…]+$")


def norm_token(value: str) -> str:
    value = str(value).strip()
    value = TAILING_DOTS_PATTERN.sub("", value)
    return value.lower()


def build_choice_set(allowed: List[str]) -> Set[str]:
    base = {norm_token(x) for x in allowed}
    base.update({"mixed", "neutral", "invalid"})
    return base


def build_context_prompt(dataset_name: str) -> str:
    if dataset_name == "imdb":
        return ("You are a movie review sentiment classifier. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    if dataset_name == "sst2":
        return ("You are a sentence-level sentiment classifier. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    if dataset_name == "imdb_sklearn":
        return ("You classify user-written movie reviews. "
                "Valid labels: positive, negative. Also allow: mixed, neutral, invalid.")
    if dataset_name == "fiqasa":
        return ("You are a financial sentiment classifier. "
                "Valid labels: positive, neutral, negative. Also allow: mixed, neutral, invalid.")
    if dataset_name == "news":
        return ("You analyze financial news headlines. "
                "Valid labels: bearish, bullish, neutral. Also allow: mixed, neutral, invalid.")
    if dataset_name == "mental":
        return ("You classify short social posts about mental health. "
                "Valid labels: normal, depression. Also allow: mixed, neutral, invalid. "
                "HARD RULE: If the text contains 'suicidal', 'suicide', or 'lifeline', "
                "classify it as 'depression'.")
    return ("Classify the text into one of the valid labels of the dataset; "
            "Also allow: mixed, neutral, invalid.")


def classify_with_gpt(dataset_name: str, pred_text: str, choices: Set[str]) -> str:
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
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"GPT 调用失败，请检查网络或 API Key：{exc}")


def direct_or_gpt(dataset_name: str, pred_text: str, choices: Set[str]) -> tuple[str, bool]:
    token = norm_token(pred_text)
    if token in choices:
        return token, False
    return classify_with_gpt(dataset_name, pred_text, choices), True


def process_all(results_root: Path | str = "results", model_dirs: List[str] | None = None):
    results_root = Path(results_root)
    entries = ordered_sentiment_entries(results_root)
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")
    if model_dirs is None:
        model_dirs = [entry["display_name"] for entry in entries]

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_file = ds["file"]
        allowed = ds["allowed"]
        choices = build_choice_set(allowed)
        base_dir = resolve_dataset_base(results_root, ds["base_path"])

        print(f"\n============================")
        print(f"📚 Dataset: {ds_name}")
        print(f"📝 Allowed: {allowed} | Extra allowed: ['mixed','neutral','invalid']")

        for model_dir in model_dirs:
            folder = base_dir / model_dir
            invalid_path = folder / ds_file.replace(".csv", ".invalid.csv")
            labeled_path = folder / ds_file.replace(".csv", ".invalid.labeled.csv")
            relabeled_path = folder / ds_file.replace(".csv", ".relabeled.csv")

            if relabeled_path.exists():
                print(f"  ⏭️ 已有 relabeled 文件，跳过：{relabeled_path}")
                continue
            if labeled_path.exists():
                print(f"  ⏭️ 已有 labeled 文件，跳过：{labeled_path}")
                continue
            if not invalid_path.exists():
                print(f"  ⚠️ Missing invalid file：{invalid_path}")
                continue

            print(f"  📄 Processing invalids -> {model_dir}")
            df = pd.read_csv(invalid_path)
            if "prediction" not in df.columns:
                print(f"  ⚠️ 'prediction' column not found in {invalid_path}")
                continue

            labels: List[str] = []
            gpt_calls = 0
            for i, row in df.iterrows():
                pred = str(row.get("prediction", "")).strip()
                label, used_gpt = direct_or_gpt(ds_name, pred, choices)
                labels.append(label)
                if used_gpt:
                    gpt_calls += 1
                    head = pred.replace("\n", " ")[:80]
                    print(f"   [{i+1:>5}] {head} -> {label}")
                    time.sleep(0.25)

            df["normalized_label"] = labels
            df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ Saved: {labeled_path}")
            print(f"  GPT calls: {gpt_calls}/{len(df)}")


if __name__ == "__main__":
    process_all()
