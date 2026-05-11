# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import gc
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DIMENSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("E", "I"),
    ("S", "N"),
    ("T", "F"),
    ("J", "P"),
)
DIMENSION_LABELS: dict[tuple[str, str], str] = {
    ("E", "I"): "E/I",
    ("S", "N"): "S/N",
    ("T", "F"): "T/F",
    ("J", "P"): "J/P",
}
LETTER_ORDER = ["E", "I", "S", "N", "T", "F", "J", "P"]
DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 4,
    "do_sample": False,
    "num_beams": 1,
}


@dataclass(frozen=True)
class MBTIItem:
    index: int
    question: str
    choice_a_text: str
    choice_b_text: str
    choice_a_value: str
    choice_b_value: str
    dimension: str


@dataclass(frozen=True)
class MBTIRunConfig:
    dataset_path: Path = Path("MBTI/data/MBTI_doubled_93.json")
    output_dir: Path = Path("MBTI/results/mbti_types")
    num_trials: int = 1
    decode_method: str = "logit"
    skip_invalid_pairs: bool = True
    force: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False


def _warn(message: str) -> None:
    print(f"[MBTI] WARNING: {message}", file=sys.stderr)


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", (value or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def default_dataset_path() -> Path:
    candidates = [
        Path("MBTI/data/MBTI_doubled_93.json"),
        Path("MBTI_doubled_93.json"),
        Path("datasets/MBTI_doubled_93.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _dimension_for_values(value_a: str, value_b: str) -> str | None:
    values = {value_a.upper(), value_b.upper()}
    for pair in DIMENSION_PAIRS:
        if values == set(pair):
            return DIMENSION_LABELS[pair]
    return None


def load_mbti_items(dataset_path: Path, skip_invalid_pairs: bool = True) -> tuple[list[MBTIItem], list[dict]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"MBTI dataset not found: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    items: list[MBTIItem] = []
    skipped: list[dict] = []
    for idx, row in enumerate(data):
        choice_a = row.get("choice_a", {})
        choice_b = row.get("choice_b", {})
        value_a = str(choice_a.get("value", "")).strip().upper()
        value_b = str(choice_b.get("value", "")).strip().upper()
        dimension = _dimension_for_values(value_a, value_b)
        if dimension is None:
            skipped.append(
                {
                    "index": idx,
                    "question": row.get("question", ""),
                    "choice_a_value": value_a,
                    "choice_b_value": value_b,
                    "reason": "choices_are_not_opposite_values_in_one_mbti_dimension",
                }
            )
            if skip_invalid_pairs:
                continue
            dimension = "cross"

        items.append(
            MBTIItem(
                index=idx,
                question=str(row.get("question", "")),
                choice_a_text=str(choice_a.get("text", "")),
                choice_b_text=str(choice_b.get("text", "")),
                choice_a_value=value_a,
                choice_b_value=value_b,
                dimension=dimension,
            )
        )
    if skipped:
        action = "Skipping" if skip_invalid_pairs else "Keeping"
        _warn(f"{action} {len(skipped)} cross-dimension MBTI items from {dataset_path}.")
    return items, skipped


def _build_prompt(item: MBTIItem) -> str:
    return (
        "You are answering a personality test.\n"
        "Choose the option that is more suitable for you.\n\n"
        f"Question: {item.question}\n"
        f"A. {item.choice_a_text}\n"
        f"B. {item.choice_b_text}\n\n"
        "Reply with only one letter: A or B."
    )


def _format_prompt(tokenizer, user_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def _single_token_candidate_ids(tokenizer, letter: str) -> set[int]:
    forms = [
        letter.lower(),
        letter.upper(),
        f" {letter.lower()}",
        f" {letter.upper()}",
        f"\n{letter.lower()}",
        f"\n{letter.upper()}",
    ]
    ids: set[int] = set()
    for text in forms:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            ids.add(int(token_ids[0]))
    return ids


def _decode_by_next_token_logits(prompt: str, tokenizer, model) -> tuple[str | None, float | None, float | None]:
    import torch

    full_prompt = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    a_ids = _single_token_candidate_ids(tokenizer, "a")
    b_ids = _single_token_candidate_ids(tokenizer, "b")
    if not a_ids or not b_ids:
        return None, None, None

    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]
    a_score = float(torch.max(logits[list(a_ids)]).detach().cpu())
    b_score = float(torch.max(logits[list(b_ids)]).detach().cpu())
    answer = "a" if a_score >= b_score else "b"
    return answer, a_score, b_score


def _parse_generated_choice(text: str) -> str | None:
    stripped = (text or "").strip().lower()
    match = re.search(r"\b([ab])\b", stripped)
    if match:
        return match.group(1)
    if stripped[:1] in {"a", "b"}:
        return stripped[:1]
    return None


def _decode_by_greedy_generation(prompt: str, tokenizer, model) -> tuple[str | None, float | None, float | None]:
    import torch

    full_prompt = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    kwargs = dict(DEFAULT_GENERATION_KWARGS)
    kwargs["eos_token_id"] = tokenizer.eos_token_id
    kwargs["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model.generate(**inputs, **kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return _parse_generated_choice(decoded), None, None


def _predict_choice(prompt: str, tokenizer, model, decode_method: str) -> tuple[str | None, float | None, float | None]:
    method = decode_method.lower().strip()
    if method == "logit":
        answer, a_score, b_score = _decode_by_next_token_logits(prompt, tokenizer, model)
        if answer is not None:
            return answer, a_score, b_score
        _warn("No single-token A/B candidates found for tokenizer; falling back to greedy generation.")
        return _decode_by_greedy_generation(prompt, tokenizer, model)
    if method == "generate":
        return _decode_by_greedy_generation(prompt, tokenizer, model)
    raise ValueError(f"Unsupported MBTI decode method: {decode_method}. Use 'logit' or 'generate'.")


def _winner(left: str, right: str, scores: Counter) -> str:
    return left if scores[left] >= scores[right] else right


def _mbti_type_from_scores(scores: Counter) -> str:
    return "".join(_winner(left, right, scores) for left, right in DIMENSION_PAIRS)


def _dimension_score_rows(model_name: str, scores: Counter) -> list[dict]:
    rows: list[dict] = []
    for left, right in DIMENSION_PAIRS:
        total = int(scores[left] + scores[right])
        rows.append(
            {
                "model": model_name,
                "dimension": DIMENSION_LABELS[(left, right)],
                "left_letter": left,
                "right_letter": right,
                "left_score": int(scores[left]),
                "right_score": int(scores[right]),
                "left_ratio": float(scores[left] / total) if total else math.nan,
                "right_ratio": float(scores[right] / total) if total else math.nan,
                "winner": _winner(left, right, scores) if total else "",
                "margin": int(abs(scores[left] - scores[right])),
                "total": total,
            }
        )
    return rows


def _load_tokenizer_and_model(model_path: Path, config: MBTIRunConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = config.torch_dtype.lower().strip()
    if dtype_name == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif dtype_name == "float16":
        torch_dtype = torch.float16
    elif dtype_name == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_name == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported torch dtype for MBTI: {config.torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    ).eval()
    return tokenizer, model


def _write_model_report(
    save_dir: Path,
    model_name: str,
    model_path: Path,
    trial_rows: list[dict],
    dimension_rows: list[dict],
    item_rows: list[dict],
    skipped_items: list[dict],
    summary_row: dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(save_dir / "mbti_trial_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dimension_rows).to_csv(save_dir / "mbti_dimension_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(item_rows).to_csv(save_dir / "mbti_item_predictions.csv", index=False, encoding="utf-8-sig")
    if skipped_items:
        pd.DataFrame(skipped_items).to_csv(save_dir / "mbti_skipped_items.csv", index=False, encoding="utf-8-sig")

    with (save_dir / "mbti_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_row,
                "dimension_scores": dimension_rows,
                "trials": trial_rows,
                "skipped_items": skipped_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    type_counter = Counter(str(row["mbti_type"]) for row in trial_rows)
    lines = [
        f"Model: {model_name}",
        f"Path: {model_path}",
        f"Decode method: {summary_row['decode_method']}",
        f"Trials completed: {summary_row['num_trials_completed']}",
        f"Items used per trial: {summary_row['num_items_used']}",
        f"Skipped invalid items: {summary_row['num_items_skipped']}",
        "",
        "Dimension scores (summed across trials):",
    ]
    for row in dimension_rows:
        lines.append(
            f"- {row['dimension']}: {row['left_letter']}={row['left_score']}, "
            f"{row['right_letter']}={row['right_score']} -> {row['winner']} "
            f"(margin={row['margin']})"
        )
    lines.extend(
        [
            "",
            f"Final MBTI by summed dimension scores: {summary_row['final_mbti_type']}",
            f"Most common per-trial MBTI: {summary_row['most_common_trial_type']}",
            f"Per-trial MBTI counts: {dict(type_counter)}",
            "",
            "Per-trial scores:",
        ]
    )
    for row in trial_rows:
        score_text = ", ".join(f"{letter}={row[f'score_{letter}']}" for letter in LETTER_ORDER)
        lines.append(f"- trial {row['trial']}: {row['mbti_type']} ({score_text})")
    (save_dir / "final_mbti_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_single_model(
    model_name: str,
    model_path: Path | str,
    items: Sequence[MBTIItem],
    skipped_items: list[dict],
    config: MBTIRunConfig,
) -> dict | None:
    model_path = Path(model_path).expanduser()
    save_dir = config.output_dir / _sanitize_filename(model_name)
    existing_summary = save_dir / "mbti_summary.json"
    if existing_summary.exists() and not config.force:
        _warn(f"MBTI result already exists for {model_name}; skipping. Use --mbti-force to rerun.")
        try:
            return json.loads(existing_summary.read_text(encoding="utf-8"))["summary"]
        except Exception:
            return None
    if not model_path.exists():
        _warn(f"Model path does not exist for {model_name}: {model_path}; skipping MBTI.")
        return None

    print(f"\n[MBTI] Evaluating {model_name}: {model_path}")
    tokenizer, model = _load_tokenizer_and_model(model_path, config)

    aggregate_scores = Counter({letter: 0 for letter in LETTER_ORDER})
    trial_rows: list[dict] = []
    item_rows: list[dict] = []

    for trial_idx in range(1, config.num_trials + 1):
        scores = Counter({letter: 0 for letter in LETTER_ORDER})
        valid_predictions = 0
        invalid_predictions = 0
        for item in items:
            prompt = _build_prompt(item)
            answer, a_score, b_score = _predict_choice(prompt, tokenizer, model, config.decode_method)
            chosen_value = ""
            if answer == "a":
                chosen_value = item.choice_a_value
            elif answer == "b":
                chosen_value = item.choice_b_value

            valid = chosen_value in LETTER_ORDER
            if valid:
                scores[chosen_value] += 1
                aggregate_scores[chosen_value] += 1
                valid_predictions += 1
            else:
                invalid_predictions += 1

            item_rows.append(
                {
                    "model": model_name,
                    "trial": trial_idx,
                    "item_index": item.index,
                    "dimension": item.dimension,
                    "question": item.question,
                    "choice_a_value": item.choice_a_value,
                    "choice_a_text": item.choice_a_text,
                    "choice_b_value": item.choice_b_value,
                    "choice_b_text": item.choice_b_text,
                    "answer": answer or "",
                    "chosen_value": chosen_value,
                    "valid": bool(valid),
                    "a_score": a_score,
                    "b_score": b_score,
                    "score_margin_a_minus_b": (a_score - b_score) if a_score is not None and b_score is not None else math.nan,
                }
            )

        mbti_type = _mbti_type_from_scores(scores)
        trial_row = {
            "model": model_name,
            "trial": trial_idx,
            "mbti_type": mbti_type,
            "valid_predictions": valid_predictions,
            "invalid_predictions": invalid_predictions,
        }
        for letter in LETTER_ORDER:
            trial_row[f"score_{letter}"] = int(scores[letter])
        for row in _dimension_score_rows(model_name, scores):
            label = row["dimension"].replace("/", "")
            trial_row[f"{label}_winner"] = row["winner"]
            trial_row[f"{label}_margin"] = row["margin"]
        trial_rows.append(trial_row)
        print(f"[MBTI] {model_name} trial {trial_idx}/{config.num_trials}: {mbti_type}")

    final_type = _mbti_type_from_scores(aggregate_scores)
    most_common_trial_type = Counter(str(row["mbti_type"]) for row in trial_rows).most_common(1)
    dimension_rows = _dimension_score_rows(model_name, aggregate_scores)

    summary_row: dict = {
        "model": model_name,
        "model_path": str(model_path),
        "dataset_path": str(config.dataset_path),
        "decode_method": config.decode_method,
        "num_trials_completed": len(trial_rows),
        "num_items_total": len(items) + len(skipped_items),
        "num_items_used": len(items),
        "num_items_skipped": len(skipped_items),
        "valid_predictions": int(sum(row["valid_predictions"] for row in trial_rows)),
        "invalid_predictions": int(sum(row["invalid_predictions"] for row in trial_rows)),
        "final_mbti_type": final_type,
        "most_common_trial_type": most_common_trial_type[0][0] if most_common_trial_type else "",
        "most_common_trial_type_count": most_common_trial_type[0][1] if most_common_trial_type else 0,
    }
    for letter in LETTER_ORDER:
        summary_row[f"score_{letter}"] = int(aggregate_scores[letter])
        summary_row[f"mean_score_{letter}"] = float(aggregate_scores[letter] / len(trial_rows)) if trial_rows else math.nan
    for row in dimension_rows:
        label = row["dimension"].replace("/", "")
        summary_row[f"{label}_winner"] = row["winner"]
        summary_row[f"{label}_margin"] = row["margin"]
        summary_row[f"{label}_left_score"] = row["left_score"]
        summary_row[f"{label}_right_score"] = row["right_score"]

    _write_model_report(
        save_dir=save_dir,
        model_name=model_name,
        model_path=model_path,
        trial_rows=trial_rows,
        dimension_rows=dimension_rows,
        item_rows=item_rows,
        skipped_items=skipped_items,
        summary_row=summary_row,
    )

    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return summary_row


def run_mbti_for_model_specs(model_specs: Sequence[dict], config: MBTIRunConfig) -> list[dict]:
    if config.num_trials < 1:
        raise ValueError("MBTI num_trials must be >= 1.")
    config = MBTIRunConfig(
        dataset_path=config.dataset_path,
        output_dir=config.output_dir,
        num_trials=config.num_trials,
        decode_method=config.decode_method,
        skip_invalid_pairs=config.skip_invalid_pairs,
        force=config.force,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    items, skipped_items = load_mbti_items(config.dataset_path, skip_invalid_pairs=config.skip_invalid_pairs)
    if not items:
        raise ValueError(f"No usable MBTI items loaded from {config.dataset_path}")

    summary_rows: list[dict] = []
    for spec in model_specs:
        model_name = str(spec.get("display_name") or spec.get("name") or spec.get("model") or "model")
        model_path = spec.get("checkpoint_path") or spec.get("path") or spec.get("model_path")
        if not model_path:
            _warn(f"Model spec has no checkpoint_path for {model_name}; skipping MBTI.")
            continue
        row = evaluate_single_model(model_name, Path(model_path), items, skipped_items, config)
        if row is not None:
            summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(config.output_dir / "mbti_summary.csv", index=False, encoding="utf-8-sig")
    else:
        _warn("No MBTI summaries were produced.")
    return summary_rows


def path_looks_like_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    marker_names = {
        "config.json",
        "adapter_config.json",
        "tokenizer_config.json",
        "generation_config.json",
    }
    if any((path / name).exists() for name in marker_names):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def discover_newlayout_checkpoint_specs(model_root: Path) -> list[dict]:
    specs: list[dict] = []
    base_path = model_root / "base"
    if path_looks_like_checkpoint(base_path):
        specs.append({"display_name": "BASE", "checkpoint_path": str(base_path)})

    reserved = {"base", "plots", "summaries", "statistical_analysis", "meta", "mbti", "__pycache__"}
    for dimension_root in sorted(p for p in model_root.iterdir() if p.is_dir() and p.name not in reserved):
        for child in sorted(p for p in dimension_root.iterdir() if p.is_dir()):
            if child.name in reserved:
                continue
            if path_looks_like_checkpoint(child):
                specs.append({"display_name": child.name.upper(), "checkpoint_path": str(child)})
    return specs


def normalize_model_specs(model_specs: Iterable[tuple[str, str] | dict]) -> list[dict]:
    normalized: list[dict] = []
    for spec in model_specs:
        if isinstance(spec, dict):
            normalized.append(dict(spec))
        else:
            name, path = spec
            normalized.append({"display_name": name, "checkpoint_path": path})
    return normalized
