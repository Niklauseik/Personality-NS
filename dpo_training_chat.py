# hf dpo suggested format (refactored for pipeline use)
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from pipeline_utils import opposite_preferred_subtype

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

RAW_DIR = Path("datasets/training_raw")  # 直接用最原始 json


def _load_raw(dimension: str, subtype: str) -> List[dict]:
    fp = RAW_DIR / f"en_{dimension}_{subtype}.json"
    if not fp.exists():
        raise FileNotFoundError(f"未找到原始数据文件 {fp}")
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"原始数据需为列表 {fp}")
    return data


def _make_user_content(instruction: str, in_ctx: str) -> str:
    instruction = (instruction or "").strip()
    in_ctx = (in_ctx or "").strip()
    if in_ctx:
        return f"{instruction}\n\n[Additional Context]\n{in_ctx}"
    return instruction


def _build_dpo_samples(tokenizer, dimension: str, preferred: str) -> List[Dict]:
    """
    从原始 json 构建 prompt / chosen / rejected 三列，使用完整数据集。
    """
    other = opposite_preferred_subtype(dimension, preferred)
    data_pref = _load_raw(dimension, preferred)
    data_other = _load_raw(dimension, other)

    n = min(len(data_pref), len(data_other))

    samples: List[Dict] = []
    for i in range(n):
        p = data_pref[i]
        q = data_other[i]
        inst_p, inp_p, out_p = p.get("instruction", ""), p.get("input", ""), p.get("output", "")
        out_q = q.get("output", "")

        if not inst_p or not out_p or not out_q:
            continue

        user_text = _make_user_content(inst_p, inp_p)
        messages = [{"role": "user", "content": user_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        samples.append(
            {
                "prompt": prompt_text,
                "chosen": (out_p or "").strip(),
                "rejected": (out_q or "").strip(),
            }
        )

    return samples


def _dataset_from_samples(samples: List[Dict]) -> Dataset:
    if not samples:
        raise RuntimeError("构建后的样本为空，请检查原始数据内容")
    return Dataset.from_list(samples)


def _build_dpo_dataset_from_raw(tokenizer, dimension: str, preferred: str) -> Dataset:
    samples = _build_dpo_samples(tokenizer, dimension, preferred)
    return _dataset_from_samples(samples)


def _build_dpo_dataset_for_sequence(
    tokenizer,
    sequence: Sequence[Dict],
) -> Dataset:
    aggregated: List[Dict] = []
    for step in sequence:
        aggregated.extend(_build_dpo_samples(tokenizer, step["dimension"], step["preferred"]))
    return _dataset_from_samples(aggregated)


def train_personality_model(
    dimension: str | None,
    preferred_subtype: str | None,
    base_model_path: str,
    save_path: str,
    personality_sequence: Sequence[Dict] | None = None,
):
    """
    Train a DPO model for either a single dimension preference or a combined personality sequence.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if personality_sequence:
        train_ds = _build_dpo_dataset_for_sequence(tokenizer, personality_sequence)
    else:
        if not dimension or not preferred_subtype:
            raise ValueError("dimension/preferred_subtype 参数不能为空。")
        train_ds = _build_dpo_dataset_from_raw(
            tokenizer,
            dimension.strip().lower(),
            preferred_subtype.strip().lower(),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-5,
        beta=1.0,
        save_strategy="no",
        save_total_limit=0,
        bf16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅模型训练完成并保存至：{save_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a single DPO model for a given dimension preference.")
    parser.add_argument(
        "--dimension",
        required=True,
        choices=["energy", "information", "decision", "execution"],
        help="MBTI dimension name.",
    )
    parser.add_argument(
        "--preferred",
        required=True,
        help="Preferred subtype within the dimension (e.g., sensing, intuition, thinking).",
    )
    parser.add_argument("--base-model-path", default="./llama-3B-Instruct")
    parser.add_argument("--save-path", required=True, help="Directory to store the trained model.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train_personality_model(
        dimension=args.dimension,
        preferred_subtype=args.preferred,
        base_model_path=args.base_model_path,
        save_path=args.save_path,
    )
