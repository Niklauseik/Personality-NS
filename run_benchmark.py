# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATASETS = {
    "gsm8k_test800": "datasets/benchmark/gsm8k_test1300.csv",
    "arc_easy_test800": "datasets/benchmark/arc_easy_test2000.csv",
    "boolq_train800": "datasets/benchmark/boolq_train2000.csv",
}


def _load_datasets(dataset_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    frames = {}
    for name, csv_path in dataset_paths.items():
        frames[name] = pd.read_csv(csv_path)
    return frames


def _local_generate(prompt: str, tokenizer, model) -> str:
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _build_prompt(dataset_name: str, row: pd.Series) -> str:
    if "gsm8k" in dataset_name:
        return (
            "Solve the following math problem and output only the final number answer.\n\n"
            f"{row['question']}\n\nOnly respond with one word. Example: 8"
        )
    if "arc_easy" in dataset_name:
        return (
            "Choose the correct option (A/B/C/D) for the following question.\n\n"
            f"Question: {row['question']}\nOptions:\n{row['choices']}\n\n"
            "Only respond with one word. Example: A"
        )
    if "boolq" in dataset_name:
        return (
            "Based on the passage, answer whether the question is true or false.\n\n"
            f"Passage: {row['passage']}\n\nQuestion: {row['question']}\n\n"
            "Only respond with one word. Example: true"
        )
    return str(row.get("question", ""))


def run_benchmarks(
    model_specs: Sequence[Dict[str, str]],
    dataset_paths: Dict[str, str] | None = None,
    results_root: Path | str = "results",
) -> None:
    """
    Run the benchmark datasets for every model spec.
    Each model spec must contain: display_name, checkpoint_path.
    """
    if not model_specs:
        raise ValueError("model_specs is empty. Provide at least the base model.")

    frames = _load_datasets(dataset_paths or DEFAULT_DATASETS)
    results_root = Path(results_root)
    save_root = results_root / "benchmark"
    save_root.mkdir(parents=True, exist_ok=True)

    for spec in model_specs:
        model_name = spec["display_name"]
        model_path = spec["checkpoint_path"]
        print(f"\n==================== 🧠 正在测试模型：{model_name} ====================")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

        for dataset_name, df in frames.items():
            print(f"\n📊 正在处理数据集：{dataset_name}")
            predictions = []

            for _, row in df.iterrows():
                prompt = _build_prompt(dataset_name, row)
                try:
                    output = _local_generate(prompt, tokenizer, model)
                except Exception as exc:  # pragma: no cover
                    output = f"[Error] {exc}"
                predictions.append(output)

            df_result = df.copy()
            df_result["prediction"] = predictions
            save_dir = save_root / model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{dataset_name}_results.csv"
            df_result.to_csv(out_path, index=False, encoding="utf-8")
            print(f"✅ 结果保存到：{out_path}")

        del model
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark inference for all active models.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    from pipeline_utils import ordered_model_entries

    args = _parse_args()
    entries = ordered_model_entries(Path(args.results_root))
    if not entries:
        raise SystemExit("No pipeline metadata found. Run stage-1 pipeline first.")
    specs = [
        {"display_name": entry["display_name"], "checkpoint_path": entry["checkpoint_path"]}
        for entry in entries
    ]
    run_benchmarks(specs, results_root=args.results_root)
