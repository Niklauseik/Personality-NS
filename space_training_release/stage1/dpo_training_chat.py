"""Preference-pair construction used by the early-stop training runners."""
from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import set_seed
from common.pipeline_utils import opposite_preferred_subtype
RAW_DIR = Path("datasets/training_raw")
DEFAULT_SEED = 42
def _set_reproducible_seed(seed: int = DEFAULT_SEED) -> None:
    # Keep training reruns as stable as practical across machines.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
