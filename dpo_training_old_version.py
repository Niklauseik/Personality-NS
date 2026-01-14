from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

RAW_DIR = Path("datasets/training_raw")
DPO_DATA_DIR = Path("datasets/dpo_converted")
REQUIRED_COLUMNS = {"chosen", "rejected"}

MBTI_DIMENSIONS = {
    0: ("E", "I", "energy_extraversion", "energy_introversion"),
    1: ("N", "S", "information_intuition", "information_sensing"),
    2: ("T", "F", "decision_thinking", "decision_feeling"),
    3: ("J", "P", "execution_judging", "execution_perceiving"),
}


def build_dpo_csv_for_dimension(
    dim_id: int,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DPO_DATA_DIR,
) -> tuple[Path, Path]:
    _, _, file_a, file_b = MBTI_DIMENSIONS[dim_id]
    path_a = raw_dir / f"en_{file_a}.json"
    path_b = raw_dir / f"en_{file_b}.json"

    with path_a.open("r", encoding="utf-8") as f:
        data_a = json.load(f)
    with path_b.open("r", encoding="utf-8") as f:
        data_b = json.load(f)

    count = min(len(data_a), len(data_b))

    records_a: List[Dict] = []
    records_b: List[Dict] = []

    for i in range(count):
        inst_a = (data_a[i].get("instruction", "") or "").strip()
        inst_b = (data_b[i].get("instruction", "") or "").strip()
        out_a = (data_a[i].get("output", "") or "").strip()
        out_b = (data_b[i].get("output", "") or "").strip()

        if not inst_a or not out_a or not out_b:
            continue

        prompt_b = inst_b if inst_b else inst_a

        records_a.append(
            {
                "chosen": json.dumps(
                    [
                        {"role": "user", "content": inst_a},
                        {"role": "assistant", "content": out_a},
                    ],
                    ensure_ascii=False,
                ),
                "rejected": json.dumps(
                    [
                        {"role": "user", "content": inst_a},
                        {"role": "assistant", "content": out_b},
                    ],
                    ensure_ascii=False,
                ),
                "score_chosen": 8,
                "score_rejected": 1,
            }
        )

        records_b.append(
            {
                "chosen": json.dumps(
                    [
                        {"role": "user", "content": prompt_b},
                        {"role": "assistant", "content": out_b},
                    ],
                    ensure_ascii=False,
                ),
                "rejected": json.dumps(
                    [
                        {"role": "user", "content": prompt_b},
                        {"role": "assistant", "content": out_a},
                    ],
                    ensure_ascii=False,
                ),
                "score_chosen": 8,
                "score_rejected": 1,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path_a = output_dir / f"{file_a}_dpo.csv"
    out_path_b = output_dir / f"{file_b}_dpo.csv"

    pd.DataFrame(records_a).to_csv(out_path_a, index=False, encoding="utf-8-sig")
    pd.DataFrame(records_b).to_csv(out_path_b, index=False, encoding="utf-8-sig")

    print(f"Built {file_a}: {len(records_a)} rows -> {out_path_a}")
    print(f"Built {file_b}: {len(records_b)} rows -> {out_path_b}")

    return out_path_a, out_path_b


def build_all_dpo_csvs(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DPO_DATA_DIR,
) -> List[Path]:
    outputs: List[Path] = []
    for dim_id in sorted(MBTI_DIMENSIONS.keys()):
        outputs.extend(build_dpo_csv_for_dimension(dim_id, raw_dir=raw_dir, output_dir=output_dir))
    return outputs


def _ensure_dpo_csvs(data_dir: Path, raw_dir: Path) -> None:
    expected = []
    for _, _, file_a, file_b in MBTI_DIMENSIONS.values():
        expected.append(data_dir / f"{file_a}_dpo.csv")
        expected.append(data_dir / f"{file_b}_dpo.csv")
    if any(not path.exists() for path in expected):
        print("DPO CSVs missing; building from raw JSON...")
        build_all_dpo_csvs(raw_dir=raw_dir, output_dir=data_dir)


def _csv_path(dimension: str, preferred: str, data_dir: Path) -> Path:
    dim = (dimension or "").strip().lower()
    pref = (preferred or "").strip().lower()
    if not dim or not pref:
        raise ValueError("dimension/preferred_subtype must be provided.")
    return data_dir / f"{dim}_{pref}_dpo.csv"


def _load_csv_dataset(csv_path: Path):
    from datasets import load_dataset

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing DPO CSV: {csv_path}. Build the converted dataset first."
        )
    ds = load_dataset("csv", data_files={"train": str(csv_path)})["train"]
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"DPO CSV {csv_path} is missing columns: {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}."
        )
    if "prompt" not in ds.column_names:
        ds = ds.add_column("prompt", [""] * len(ds))
    return ds


def _dataset_for_sequence(sequence: Sequence[Dict], data_dir: Path):
    from datasets import concatenate_datasets

    datasets: List = []
    for step in sequence:
        csv_path = _csv_path(step["dimension"], step["preferred"], data_dir)
        datasets.append(_load_csv_dataset(csv_path))
    if not datasets:
        raise RuntimeError("No datasets loaded for the requested sequence.")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _train_from_dataset(train_ds, base_model_path: str, save_path: str) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
        num_train_epochs=6,
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

    print(f"\nModel training complete. Saved to: {save_path}")


def train_personality_model(
    dimension: str | None,
    preferred_subtype: str | None,
    base_model_path: str,
    save_path: str,
    personality_sequence: Sequence[Dict] | None = None,
    data_dir: str | Path = DPO_DATA_DIR,
) -> None:
    data_dir = Path(data_dir)
    _ensure_dpo_csvs(data_dir=data_dir, raw_dir=RAW_DIR)

    if personality_sequence:
        train_ds = _dataset_for_sequence(personality_sequence, data_dir)
    else:
        if not dimension or not preferred_subtype:
            raise ValueError("dimension/preferred_subtype must be provided.")
        csv_path = _csv_path(dimension, preferred_subtype, data_dir)
        train_ds = _load_csv_dataset(csv_path)

    _train_from_dataset(train_ds, base_model_path=base_model_path, save_path=save_path)


def train_dpo_model(data_path: str, save_path: str, base_model_path: str = "./llama-3B-Instruct") -> None:
    train_ds = _load_csv_dataset(Path(data_path))
    _train_from_dataset(train_ds, base_model_path=base_model_path, save_path=save_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DPO model using CSV data (old version format)."
    )
    parser.add_argument(
        "--dimension",
        choices=["energy", "information", "decision", "execution"],
        help="MBTI dimension name.",
    )
    parser.add_argument(
        "--preferred",
        help="Preferred subtype within the dimension (e.g., sensing, intuition, thinking).",
    )
    parser.add_argument("--base-model-path", default="./llama-3B-Instruct")
    parser.add_argument("--save-path", help="Directory to store the trained model.")
    parser.add_argument(
        "--dpo-data-dir",
        default=str(DPO_DATA_DIR),
        help="Directory containing converted DPO CSV files.",
    )
    parser.add_argument(
        "--build-dpo-csvs",
        action="store_true",
        help="Build all DPO CSVs from raw JSON and exit.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.build_dpo_csvs:
        build_all_dpo_csvs(raw_dir=RAW_DIR, output_dir=Path(args.dpo_data_dir))
        raise SystemExit(0)
    if not args.dimension or not args.preferred or not args.save_path:
        raise SystemExit("Missing required arguments: --dimension --preferred --save-path")
    train_personality_model(
        dimension=args.dimension,
        preferred_subtype=args.preferred,
        base_model_path=args.base_model_path,
        save_path=args.save_path,
        data_dir=args.dpo_data_dir,
    )
