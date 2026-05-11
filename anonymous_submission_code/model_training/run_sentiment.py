# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_CONFIGS = {
    "imdb": "datasets/sentiment/imdb.csv",
    "sst2": "datasets/sentiment/sst2.csv",
    "imdb_sklearn": "datasets/sentiment/imdb_sklearn.csv",
    "fiqasa": "datasets/sentiment/fiqasa.csv",
    "news": "datasets/sentiment/news_sentiment.csv",
    "mental": "datasets/sentiment/mental_health_sentiment.csv",
}


def build_prompt(dataset_name: str, text: str) -> str:
    if dataset_name == "imdb":
        return (
            "You are a movie review sentiment classifier. "
            "Classify the following review as either positive or negative. "
            "Respond with only one word: positive or negative.\n\n"
            f"Review:\n{text}\n\nSentiment:"
        )
    if dataset_name == "sst2":
        return (
            "You are a sentence-level sentiment analysis model. "
            "Classify the sentiment of the sentence as positive or negative. "
            "Respond with only one word: positive or negative.\n\n"
            f"Sentence:\n{text}\n\nSentiment:"
        )
    if dataset_name == "imdb_sklearn":
        return (
            "You are a sentiment classifier trained on user-written movie reviews. "
            "Judge whether the review is positive or negative. "
            "Respond with only one word: positive or negative.\n\n"
            f"Movie Review:\n{text}\n\nSentiment:"
        )
    if dataset_name == "fiqasa":
        return (
            "You are a financial sentiment classifier. "
            "Respond with only one word: either 'positive', 'neutral', or 'negative'.\n\n"
            f"{text}"
        )
    if dataset_name == "news":
        return (
            "You are analyzing financial news headlines. Each headline reflects a short financial opinion or fact. "
            "Classify the sentiment into one of the following categories:\n"
            "- Bearish\n- Bullish\n- Neutral\n\n"
            "Respond with one word only.\n\n"
            "Example:\n"
            "Text: $GM -- GM loses a bull\n"
            "Answer: Bearish\n\n"
            "Now classify the following:\n"
            f"{text}\nAnswer:"
        )
    if dataset_name == "mental":
        return (
            "You are given a short social media post that may reflect the mental state of the writer. "
            "Classify it as either Normal or Depression. Respond with a single word.\n\n"
            f"Text: {text}\n\nSentiment:"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _format_prompt(prompt: str, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _local_generate(prompt, tokenizer, model):
    full_prompt = _format_prompt(prompt, tokenizer)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_sentiment(
    model_specs: Sequence[Dict[str, str]],
    dataset_configs: Dict[str, str] | None = None,
    results_root: Path | str = "results",
    file_suffix: str = "",
) -> None:
    if not model_specs:
        raise ValueError("model_specs is empty. Provide at least the base model.")

    file_suffix = (file_suffix or "").strip()
    if file_suffix and not file_suffix.startswith("_"):
        file_suffix = "_" + file_suffix

    dataset_configs = dataset_configs or DATASET_CONFIGS
    results_root = Path(results_root)
    sentiment_root = results_root / "sentiment"
    sentiment_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_path in dataset_configs.items():
        df = pd.read_csv(dataset_path)
        print(f"\n📄 正在测试数据集：{dataset_name}，共 {len(df)} 条")

        for spec in model_specs:
            model_name = spec["display_name"]
            model_path = spec["checkpoint_path"]
            print(f"\n🧪 正在测试模型：{model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()

            predictions = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name} | {model_name}"):
                try:
                    prompt = build_prompt(dataset_name, row["text"])
                    pred = _local_generate(prompt, tokenizer, model)
                except Exception as exc:  # pragma: no cover
                    pred = f"[Error] {exc}"
                predictions.append(pred)

            df_result = df.copy()
            df_result["prediction"] = predictions

            save_dir = sentiment_root / dataset_name / model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{dataset_name}_sentiment_results{file_suffix}.csv"
            df_result.to_csv(out_path, index=False, encoding="utf-8")
            print(f"✅ 保存完成：{out_path}")

            del model
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.empty_cache()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment inference for all active models.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    from common.pipeline_utils import ordered_sentiment_entries

    args = _parse_args()
    entries = ordered_sentiment_entries(Path(args.results_root))
    if not entries:
        raise SystemExit("No pipeline metadata found. Run model training and inference first.")
    specs = [
        {"display_name": entry["display_name"], "checkpoint_path": entry["checkpoint_path"]}
        for entry in entries
    ]
    run_sentiment(specs, results_root=args.results_root)
