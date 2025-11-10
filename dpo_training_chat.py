# hf dpo suggested format
import os
import re
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

RAW_DIR = Path("datasets/training_raw")  # 直接用最原始 json

def _parse_dim_and_pref_from_csv_path_like(data_path: str):
    """
    兼容你原来的 data_path（如: ./datasets/dpo_converted/information_sensing_dpo.csv）
    仅用于解析出 dimension 和 preferred，不再读取 CSV。
    """
    name = Path(data_path).stem  # e.g., information_sensing_dpo
    # 提取 dimension 与 subtype
    m = re.match(r"^(energy|information|decision|execution)_(extraversion|introversion|intuition|sensing|thinking|feeling|judging|perceiving)", name)
    if not m:
        raise ValueError(f"无法从 data_path 推断维度与子类: {data_path}")
    dimension, preferred = m.group(1), m.group(2)
    return dimension, preferred

def _load_raw(dimension: str, subtype: str):
    fp = RAW_DIR / f"en_{dimension}_{subtype}.json"
    if not fp.exists():
        raise FileNotFoundError(f"未找到原始数据文件: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"原始数据需为列表: {fp}")
    return data

def _make_user_content(instruction: str, in_ctx: str) -> str:
    instruction = (instruction or "").strip()
    in_ctx = (in_ctx or "").strip()
    if in_ctx:
        return f"{instruction}\n\n[Additional Context]\n{in_ctx}"
    return instruction

def _opposite_subtype(dimension: str, subtype: str) -> str:
    pairs = {
        "decision": ("thinking", "feeling"),
        "information": ("intuition", "sensing"),
        "energy": ("extraversion", "introversion"),
        "execution": ("judging", "perceiving"),
    }
    a, b = pairs[dimension]
    return b if subtype == a else a

def _build_dpo_dataset_from_raw(tokenizer, dimension: str, preferred: str, take_n: int = 10000):
    """
    从最原始 json 构建 DPO 三列: prompt / chosen / rejected
    - prompt：对 user 部分套 apply_chat_template(add_generation_prompt=True)
    - chosen：preferred 侧的输出
    - rejected：对侧输出
    """
    other = _opposite_subtype(dimension, preferred)
    data_pref = _load_raw(dimension, preferred)
    data_other = _load_raw(dimension, other)

    n = min(len(data_pref), len(data_other), take_n)

    samples = []
    for i in range(n):
        p = data_pref[i]
        q = data_other[i]
        inst_p, inp_p, out_p = p.get("instruction",""), p.get("input",""), p.get("output","")
        out_q = q.get("output","")

        if not inst_p or not out_p or not out_q:
            continue

        user_text = _make_user_content(inst_p, inp_p)
        messages = [{"role": "user", "content": user_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        samples.append({
            "prompt": prompt_text,
            "chosen": (out_p or "").strip(),
            "rejected": (out_q or "").strip(),
        })

    if not samples:
        raise RuntimeError("构建后的样本为空，请检查原始数据内容。")

    return Dataset.from_list(samples)

def train_dpo_model(data_path: str, save_path: str):
    model_path = "./llama-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # === 从最原始数据直接构建训练集（唯一改动） ===
    dim, pref = _parse_dim_and_pref_from_csv_path_like(data_path)
    train_ds = _build_dpo_dataset_from_raw(tokenizer, dim, pref, take_n=10000)

    # === Load base model ===（保持不变）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # === Enhanced LoRA config ===（保持不变）
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === Load frozen reference model ===（保持不变，放 CPU）
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    # === DPO trainer config ===（保持不变）
    dpo_cfg = DPOConfig(
        output_dir=save_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=6,
        learning_rate=1e-5,
        beta=1.0,
        save_strategy="no",   # ✅ 不保存 checkpoint
        save_total_limit=0,    # ✅ 确保不保留历史 checkpoint
        bf16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer
    )

    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅ 模型训练完成并保存至：{save_path}")

if __name__ == "__main__":
    train_dpo_model(
        data_path="./datasets/dpo_converted/information_sensing_dpo.csv",
        save_path="./dpo_outputs/model_s_3B"
    )

    train_dpo_model(
        data_path="./datasets/dpo_converted/information_intuition_dpo.csv",
        save_path="./dpo_outputs/model_n_3B"
    )
