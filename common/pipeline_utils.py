# -*- coding: utf-8 -*-
"""
Shared helpers for the end-to-end pipeline.

This module centralizes dimension metadata, pipeline state persistence,
and convenience utilities so the stage-1 / stage-2 scripts and the
individual task modules can stay in sync.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

BASE_DISPLAY_NAME = "原始基座模型"

DIMENSION_SPECS: Dict[str, Dict] = {
    "energy": {
        "pretty": "Energy (E/I)",
        "subtypes": [
            {"code": "e", "preferred": "extraversion", "display_name": "E性格模型"},
            {"code": "i", "preferred": "introversion", "display_name": "I性格模型"},
        ],
    },
    "information": {
        "pretty": "Information (S/N)",
        "subtypes": [
            {"code": "s", "preferred": "sensing", "display_name": "S性格模型"},
            {"code": "n", "preferred": "intuition", "display_name": "N性格模型"},
        ],
    },
    "decision": {
        "pretty": "Decision (T/F)",
        "subtypes": [
            {"code": "t", "preferred": "thinking", "display_name": "T性格模型"},
            {"code": "f", "preferred": "feeling", "display_name": "F性格模型"},
        ],
    },
    "execution": {
        "pretty": "Execution (J/P)",
        "subtypes": [
            {"code": "j", "preferred": "judging", "display_name": "J性格模型"},
            {"code": "p", "preferred": "perceiving", "display_name": "P性格模型"},
        ],
    },
}

LETTER_SPECS: Dict[str, Dict] = {}
for dimension, spec in DIMENSION_SPECS.items():
    for subtype in spec["subtypes"]:
        letter = subtype["code"][0].upper()
        LETTER_SPECS[letter] = {
            "dimension": dimension,
            "preferred": subtype["preferred"],
            "display_name": subtype["display_name"],
            "code": subtype["code"],
        }


def list_allowed_dimensions() -> List[str]:
    return list(DIMENSION_SPECS.keys())


def get_dimension_spec(name: str) -> Dict:
    key = (name or "").strip().lower()
    if key not in DIMENSION_SPECS:
        raise ValueError(f"Unsupported dimension '{name}'. "
                         f"Allowed values: {', '.join(DIMENSION_SPECS.keys())}")
    return DIMENSION_SPECS[key]


def resolve_letter_spec(letter: str) -> Dict:
    key = (letter or "").strip().upper()
    if key not in LETTER_SPECS:
        raise ValueError(
            f"Unsupported letter '{letter}'. "
            "Allowed letters: " + ", ".join(sorted(LETTER_SPECS.keys()))
        )
    return dict(LETTER_SPECS[key])


def opposite_preferred_subtype(dimension: str, preferred: str) -> str:
    spec = get_dimension_spec(dimension)
    for subtype in spec["subtypes"]:
        if subtype["preferred"] != preferred:
            return subtype["preferred"]
    raise ValueError(f"Could not find opposite subtype for {dimension}/{preferred}")


def _sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", (value or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def sanitize_run_name(value: str) -> str:
    """
    Sanitize a string for safe use as a directory/file name segment.
    """
    return _sanitize_run_name(value)


def standard_model_dir(output_root: Path, base_model_path: Path | str, personality_code: str) -> Path:
    base_name = _sanitize_run_name(str(base_model_path).rstrip("/\\").split("/")[-1].split("\\")[-1])
    tag = _sanitize_run_name(personality_code)
    return output_root / f"{base_name}_{tag}性格"


def normalize_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser()


def resolve_dataset_base(results_root: Path, dataset_base: str) -> Path:
    base_path = Path(dataset_base)
    if base_path.is_absolute():
        return base_path
    parts = list(base_path.parts)
    if parts and parts[0] == "results":
        parts = parts[1:]
    return results_root.joinpath(*parts)


def get_pipeline_state_path(results_root: Path) -> Path:
    return results_root / "pipeline_state.json"


def write_pipeline_state(state: Dict, results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    path = get_pipeline_state_path(results_root)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_pipeline_state(results_root: Path) -> Dict:
    path = get_pipeline_state_path(results_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline metadata not found: {path}. "
            "Please run the stage-1 pipeline first."
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_model_entries(dimension: str,
                        base_model_path: Path,
                        trained_models: Sequence[Dict]) -> List[Dict]:
    entries = [{
        "role": "base",
        "code": "base",
        "display_name": BASE_DISPLAY_NAME,
        "checkpoint_path": str(base_model_path),
    }]
    entries.extend(trained_models)
    return entries


def generate_run_id() -> str:
    return uuid.uuid4().hex[:8]


def current_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_model_entries(results_root: Path) -> List[Dict]:
    state = load_pipeline_state(results_root)
    return state.get("model_entries", [])


def list_model_display_names(results_root: Path) -> List[str]:
    entries = load_model_entries(results_root)
    return [entry["display_name"] for entry in entries]


def list_trained_models(results_root: Path) -> List[Dict]:
    return [entry for entry in load_model_entries(results_root)
            if entry.get("role") != "base"]


def ordered_model_entries(results_root: Path) -> List[Dict]:
    """
    Returns model entries with the base model first, followed by the
    trained models in the order they were stored in the metadata.
    """
    entries = load_model_entries(results_root)
    base_entries = [e for e in entries if e.get("role") == "base"]
    others = [e for e in entries if e.get("role") != "base"]
    return base_entries + others


def ordered_sentiment_entries(results_root: Path) -> List[Dict]:
    """
    Returns the model entries that are expected to have sentiment outputs.

    When stage-1 is run with --no-base-sentiment, the pipeline metadata still
    lists the base model for benchmark comparisons, but sentiment outputs are
    only generated for trained personality models.
    """
    results_root = Path(results_root)
    state = load_pipeline_state(results_root)
    include_base = bool(state.get("base_sentiment_enabled", True))
    entries = ordered_model_entries(results_root)
    if include_base:
        return entries
    return [entry for entry in entries if entry.get("role") != "base"]


def ensure_output_target(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {path}. "
                "Please remove it manually before running the pipeline again."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)
