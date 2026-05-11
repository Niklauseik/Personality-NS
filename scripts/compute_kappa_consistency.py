#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute cross-model direction consistency with Cohen's kappa.

This script intentionally uses only the Python standard library. The current
repo environment may not have pandas/sklearn installed; the local kappa
implementation follows the unweighted Cohen's kappa definition used by
sklearn.metrics.cohen_kappa_score.

Kappa is computed over dataset-conditioned direction labels. For example, the
E-shift cell compares labels such as "imdb:+" and "mental:-" rather than
pooling all positive directions into one "+" category.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


MODEL_FAMILY_BY_ROOT = {
    "llama-3b_newlayout": "L3",
    "qwen-3b_newlayout": "Q3",
    "qwen-7b_newlayout": "Q7",
}
ROOT_BY_MODEL_FAMILY = {v: k for k, v in MODEL_FAMILY_BY_ROOT.items()}
MODEL_FAMILY_ORDER = ["L3", "Q3", "Q7"]

PERSONALITY_TO_PAIR = {
    "E": "energy",
    "I": "energy",
    "S": "information",
    "N": "information",
    "F": "decision",
    "T": "decision",
    "J": "execution",
    "P": "execution",
}
PERSONALITY_ORDER = ["E", "I", "S", "N", "F", "T", "J", "P"]
REQUESTED_PAIRS = set(PERSONALITY_TO_PAIR.values())

MODEL_PAIR_ORDER = [
    ("L3", "Q3"),
    ("L3", "Q7"),
    ("Q3", "Q7"),
]
MODEL_PAIR_LABELS = {
    ("L3", "Q3"): "L3 vs Q3",
    ("L3", "Q7"): "L3 vs Q7",
    ("Q3", "Q7"): "Q3 vs Q7",
}

DATASET_ORDER = ["imdb", "imdb_sklearn", "sst2", "fiqasa", "news", "mental"]
DATASET_DISPLAY = {
    "fiqasa": "FiQA-SA",
    "imdb": "IMDb",
    "imdb_sklearn": "IMDb-Sklearn",
    "mental": "Mental",
    "news": "News",
    "sst2": "SST-2",
}

SHIFT_AXES_BY_DATASET = {
    "imdb": ("ratio_positive", "ratio_negative"),
    "imdb_sklearn": ("ratio_positive", "ratio_negative"),
    "sst2": ("ratio_positive", "ratio_negative"),
    "fiqasa": ("ratio_positive", "ratio_negative"),
    "news": ("ratio_bullish", "ratio_bearish"),
    "mental": ("ratio_normal", "ratio_depression"),
}

WIDE_COLUMNS = [
    *(f"{personality} {metric}" for personality in PERSONALITY_ORDER for metric in ("shift", "F1")),
    "Overall shift",
    "Overall F1",
]

LATEX_CAPTION = (
    "Cross-model consistency of personality-induced behavioural shifts and performance changes. "
    "Each cell reports Cohen’s kappa between two model families, computed from dataset-conditioned "
    "direction labels for changes relative to each model’s corresponding base model. Shift and F1 "
    "denote consistency in behavioural shift direction and macro-F1 change direction, respectively."
)
LATEX_LABEL = r"\label{tab:kappa_consistency}"


@dataclass(frozen=True)
class SummaryRow:
    model_family: str
    model_root: str
    pair: str
    dataset: str
    run: str
    model: str
    source_file: str
    values: dict[str, str]


@dataclass(frozen=True)
class DirectionRow:
    model_family: str
    model_root: str
    personality: str
    pair: str
    dataset: str
    run: str
    source_file: str
    base_shift: float
    tuned_shift: float
    delta_shift: float
    shift_dir: int
    base_f1: float
    tuned_f1: float
    delta_f1: float
    f1_dir: int


def clean_fieldname(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def sort_dataset_key(dataset: str) -> tuple[int, str]:
    if dataset in DATASET_ORDER:
        return DATASET_ORDER.index(dataset), dataset
    return 10**9, dataset


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_float_raw(value: float) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.12g}"


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    value = str(value).strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def direction(delta: float, epsilon: float) -> int | None:
    if delta is None or math.isnan(delta):
        return None
    if delta > epsilon:
        return 1
    if delta < -epsilon:
        return -1
    return 0


def infer_shift_axis(dataset: str, row: SummaryRow) -> tuple[str, str] | None:
    if dataset in SHIFT_AXES_BY_DATASET:
        return SHIFT_AXES_BY_DATASET[dataset]

    labels = set(str(row.values.get("labels", "")).lower().split("|"))
    if {"positive", "negative"}.issubset(labels):
        return "ratio_positive", "ratio_negative"
    if {"bullish", "bearish"}.issubset(labels):
        return "ratio_bullish", "ratio_bearish"
    if {"normal", "depression"}.issubset(labels):
        return "ratio_normal", "ratio_depression"
    return None


def compute_shift(row: SummaryRow) -> float:
    axis = infer_shift_axis(row.dataset, row)
    if axis is None:
        return math.nan
    positive_col, negative_col = axis
    pos = parse_float(row.values.get(positive_col))
    neg = parse_float(row.values.get(negative_col))
    if math.isnan(pos) or math.isnan(neg):
        return math.nan
    return pos - neg


def cohen_kappa(labels_a: list[object], labels_b: list[object]) -> float:
    """Compute unweighted Cohen's kappa.

    If expected agreement is 1.0, kappa is undefined, matching sklearn's
    effective behavior for degenerate constant-label inputs.
    """

    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equal-length label sequences.")
    n = len(labels_a)
    if n == 0:
        return math.nan

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(counts_a[label] * counts_b[label] for label in set(counts_a) | set(counts_b)) / (n * n)
    denom = 1.0 - expected
    if abs(denom) <= 1e-15:
        return math.nan
    return (observed - expected) / denom


def direction_token(direction_label: int) -> str:
    if direction_label > 0:
        return "+"
    if direction_label < 0:
        return "-"
    return "0"


def conditioned_label(dataset: str, direction_label: int, personality: str | None = None) -> str:
    token = direction_token(direction_label)
    if personality is None:
        return f"{dataset}:{token}"
    return f"{personality}:{dataset}:{token}"


def labels_to_string(labels: Iterable[object]) -> str:
    return ",".join(str(x) for x in labels)


def read_summary_rows(root: Path) -> tuple[dict[tuple[str, str, str, str], SummaryRow], list[Path], list[Path], list[str]]:
    all_paths = sorted(root.glob("*_newlayout/*/summaries/sentiment.csv"))
    selected_paths: list[Path] = []
    skipped_paths: list[Path] = []
    warnings: list[str] = []
    rows: dict[tuple[str, str, str, str], SummaryRow] = {}

    for path in all_paths:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        parts = rel.parts
        if len(parts) < 4:
            skipped_paths.append(path)
            continue
        model_root, pair = parts[0], parts[1]
        if model_root not in MODEL_FAMILY_BY_ROOT or pair not in REQUESTED_PAIRS:
            skipped_paths.append(path)
            continue

        selected_paths.append(path)
        model_family = MODEL_FAMILY_BY_ROOT[model_root]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                warnings.append(f"WARNING empty CSV header: {rel.as_posix()}")
                continue
            reader.fieldnames = [clean_fieldname(name) for name in reader.fieldnames]
            for raw in reader:
                row = {clean_fieldname(k): v for k, v in raw.items()}
                run = str(row.get("run", "")).strip()
                if run != "avg":
                    continue
                dataset = str(row.get("dataset", "")).strip()
                model = str(row.get("model", "")).strip()
                if not dataset or not model:
                    warnings.append(f"WARNING malformed row without dataset/model in {rel.as_posix()}")
                    continue
                key = (model_family, pair, dataset, model)
                summary_row = SummaryRow(
                    model_family=model_family,
                    model_root=model_root,
                    pair=pair,
                    dataset=dataset,
                    run=run,
                    model=model,
                    source_file=rel.as_posix(),
                    values=row,
                )
                if key in rows:
                    warnings.append(f"WARNING duplicate avg row for {key}; keeping the last row from {rel.as_posix()}")
                rows[key] = summary_row

    return rows, selected_paths, skipped_paths, warnings


def build_direction_rows(
    summary_rows: dict[tuple[str, str, str, str], SummaryRow],
    datasets: list[str],
    epsilon_shift: float,
    epsilon_f1: float,
) -> tuple[list[DirectionRow], list[dict[str, str]], list[str]]:
    complete_rows: list[DirectionRow] = []
    debug_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for family in MODEL_FAMILY_ORDER:
        model_root = ROOT_BY_MODEL_FAMILY[family]
        for personality in PERSONALITY_ORDER:
            pair = PERSONALITY_TO_PAIR[personality]
            for dataset in datasets:
                base_key = (family, pair, dataset, "BASE")
                tuned_key = (family, pair, dataset, personality)
                base = summary_rows.get(base_key)
                tuned = summary_rows.get(tuned_key)

                warning_parts: list[str] = []
                if base is None:
                    warning_parts.append(f"missing base row {base_key}")
                if tuned is None:
                    warning_parts.append(f"missing tuned row {tuned_key}")

                if warning_parts:
                    warning = "; ".join(warning_parts)
                    warnings.append(f"WARNING {family}-{personality}-{dataset}: {warning}")
                    debug_rows.append(
                        {
                            "model_family": family,
                            "model_root": model_root,
                            "personality": personality,
                            "pair": pair,
                            "dataset": dataset,
                            "run": "avg",
                            "source_file": "",
                            "base_shift": "",
                            "tuned_shift": "",
                            "delta_shift": "",
                            "shift_dir": "",
                            "base_f1": "",
                            "tuned_f1": "",
                            "delta_f1": "",
                            "f1_dir": "",
                            "is_complete": "false",
                            "warning": warning,
                        }
                    )
                    continue

                assert base is not None and tuned is not None
                base_shift = compute_shift(base)
                tuned_shift = compute_shift(tuned)
                base_f1 = parse_float(base.values.get("f1_macro_strict"))
                tuned_f1 = parse_float(tuned.values.get("f1_macro_strict"))

                if math.isnan(base_shift) or math.isnan(tuned_shift):
                    warning_parts.append("missing or invalid shift axis value")
                if math.isnan(base_f1) or math.isnan(tuned_f1):
                    warning_parts.append("missing or invalid f1_macro_strict")

                if warning_parts:
                    warning = "; ".join(warning_parts)
                    warnings.append(f"WARNING {family}-{personality}-{dataset}: {warning}")
                    debug_rows.append(
                        {
                            "model_family": family,
                            "model_root": model_root,
                            "personality": personality,
                            "pair": pair,
                            "dataset": dataset,
                            "run": "avg",
                            "source_file": tuned.source_file,
                            "base_shift": fmt_float_raw(base_shift),
                            "tuned_shift": fmt_float_raw(tuned_shift),
                            "delta_shift": "",
                            "shift_dir": "",
                            "base_f1": fmt_float_raw(base_f1),
                            "tuned_f1": fmt_float_raw(tuned_f1),
                            "delta_f1": "",
                            "f1_dir": "",
                            "is_complete": "false",
                            "warning": warning,
                        }
                    )
                    continue

                delta_shift = tuned_shift - base_shift
                delta_f1 = tuned_f1 - base_f1
                shift_dir = direction(delta_shift, epsilon_shift)
                f1_dir = direction(delta_f1, epsilon_f1)
                if shift_dir is None or f1_dir is None:
                    warning = "unable to assign direction label"
                    warnings.append(f"WARNING {family}-{personality}-{dataset}: {warning}")
                    continue

                row = DirectionRow(
                    model_family=family,
                    model_root=model_root,
                    personality=personality,
                    pair=pair,
                    dataset=dataset,
                    run="avg",
                    source_file=tuned.source_file,
                    base_shift=base_shift,
                    tuned_shift=tuned_shift,
                    delta_shift=delta_shift,
                    shift_dir=shift_dir,
                    base_f1=base_f1,
                    tuned_f1=tuned_f1,
                    delta_f1=delta_f1,
                    f1_dir=f1_dir,
                )
                complete_rows.append(row)
                debug_rows.append(
                    {
                        "model_family": row.model_family,
                        "model_root": row.model_root,
                        "personality": row.personality,
                        "pair": row.pair,
                        "dataset": row.dataset,
                        "run": row.run,
                        "source_file": row.source_file,
                        "base_shift": fmt_float_raw(row.base_shift),
                        "tuned_shift": fmt_float_raw(row.tuned_shift),
                        "delta_shift": fmt_float_raw(row.delta_shift),
                        "shift_dir": str(row.shift_dir),
                        "base_f1": fmt_float_raw(row.base_f1),
                        "tuned_f1": fmt_float_raw(row.tuned_f1),
                        "delta_f1": fmt_float_raw(row.delta_f1),
                        "f1_dir": str(row.f1_dir),
                        "is_complete": "true",
                        "warning": "",
                    }
                )

    return complete_rows, debug_rows, warnings


def compute_kappas(
    direction_rows: list[DirectionRow],
    warnings: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, str]], dict[str, dict[str, int]]]:
    by_key = {
        (row.model_family, row.personality, row.dataset): row
        for row in direction_rows
    }
    available_datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in direction_rows:
        available_datasets[(row.model_family, row.personality)].add(row.dataset)

    long_rows: list[dict[str, object]] = []
    wide_rows_by_pair: dict[str, dict[str, str]] = {}
    datapoints_by_pair: dict[str, dict[str, int]] = {}

    for family_a, family_b in MODEL_PAIR_ORDER:
        pair_label = MODEL_PAIR_LABELS[(family_a, family_b)]
        wide_row: dict[str, str] = {"model_pair": pair_label}
        overall_shift_a: list[str] = []
        overall_shift_b: list[str] = []
        overall_f1_a: list[str] = []
        overall_f1_b: list[str] = []
        overall_dataset_tokens: list[str] = []

        for personality in PERSONALITY_ORDER:
            datasets_a = available_datasets.get((family_a, personality), set())
            datasets_b = available_datasets.get((family_b, personality), set())
            common = sorted(datasets_a & datasets_b, key=sort_dataset_key)
            union = sorted(datasets_a | datasets_b, key=sort_dataset_key)
            if len(common) != len(union):
                missing = sorted(set(union) - set(common), key=sort_dataset_key)
                warnings.append(
                    f"WARNING {pair_label} {personality}: using {len(common)} common datasets; "
                    f"missing on one side: {', '.join(missing)}"
                )

            shift_a = [
                conditioned_label(dataset, by_key[(family_a, personality, dataset)].shift_dir)
                for dataset in common
            ]
            shift_b = [
                conditioned_label(dataset, by_key[(family_b, personality, dataset)].shift_dir)
                for dataset in common
            ]
            f1_a = [
                conditioned_label(dataset, by_key[(family_a, personality, dataset)].f1_dir)
                for dataset in common
            ]
            f1_b = [
                conditioned_label(dataset, by_key[(family_b, personality, dataset)].f1_dir)
                for dataset in common
            ]

            shift_kappa = cohen_kappa(shift_a, shift_b)
            f1_kappa = cohen_kappa(f1_a, f1_b)
            if math.isnan(shift_kappa):
                warnings.append(
                    f"WARNING {pair_label} {personality} shift: Cohen's kappa is undefined "
                    "because the direction labels have no effective variation."
                )
            if math.isnan(f1_kappa):
                warnings.append(
                    f"WARNING {pair_label} {personality} F1: Cohen's kappa is undefined "
                    "because the direction labels have no effective variation."
                )
            wide_row[f"{personality} shift"] = fmt_float(shift_kappa)
            wide_row[f"{personality} F1"] = fmt_float(f1_kappa)

            datasets_text = ",".join(common)
            long_rows.append(
                {
                    "model_pair": pair_label,
                    "personality": personality,
                    "metric": "shift",
                    "kappa": shift_kappa,
                    "n_datasets": len(common),
                    "labels_a": labels_to_string(shift_a),
                    "labels_b": labels_to_string(shift_b),
                    "datasets": datasets_text,
                }
            )
            long_rows.append(
                {
                    "model_pair": pair_label,
                    "personality": personality,
                    "metric": "F1",
                    "kappa": f1_kappa,
                    "n_datasets": len(common),
                    "labels_a": labels_to_string(f1_a),
                    "labels_b": labels_to_string(f1_b),
                    "datasets": datasets_text,
                }
            )

            overall_shift_a.extend(
                conditioned_label(dataset, by_key[(family_a, personality, dataset)].shift_dir, personality)
                for dataset in common
            )
            overall_shift_b.extend(
                conditioned_label(dataset, by_key[(family_b, personality, dataset)].shift_dir, personality)
                for dataset in common
            )
            overall_f1_a.extend(
                conditioned_label(dataset, by_key[(family_a, personality, dataset)].f1_dir, personality)
                for dataset in common
            )
            overall_f1_b.extend(
                conditioned_label(dataset, by_key[(family_b, personality, dataset)].f1_dir, personality)
                for dataset in common
            )
            overall_dataset_tokens.extend([f"{personality}:{dataset}" for dataset in common])

        overall_shift = cohen_kappa(overall_shift_a, overall_shift_b)
        overall_f1 = cohen_kappa(overall_f1_a, overall_f1_b)
        if math.isnan(overall_shift):
            warnings.append(
                f"WARNING {pair_label} Overall shift: Cohen's kappa is undefined "
                "because the direction labels have no effective variation."
            )
        if math.isnan(overall_f1):
            warnings.append(
                f"WARNING {pair_label} Overall F1: Cohen's kappa is undefined "
                "because the direction labels have no effective variation."
            )
        wide_row["Overall shift"] = fmt_float(overall_shift)
        wide_row["Overall F1"] = fmt_float(overall_f1)

        long_rows.append(
            {
                "model_pair": pair_label,
                "personality": "Overall",
                "metric": "shift",
                "kappa": overall_shift,
                "n_datasets": len(overall_shift_a),
                "labels_a": labels_to_string(overall_shift_a),
                "labels_b": labels_to_string(overall_shift_b),
                "datasets": ",".join(overall_dataset_tokens),
            }
        )
        long_rows.append(
            {
                "model_pair": pair_label,
                "personality": "Overall",
                "metric": "F1",
                "kappa": overall_f1,
                "n_datasets": len(overall_f1_a),
                "labels_a": labels_to_string(overall_f1_a),
                "labels_b": labels_to_string(overall_f1_b),
                "datasets": ",".join(overall_dataset_tokens),
            }
        )
        wide_rows_by_pair[pair_label] = wide_row
        datapoints_by_pair[pair_label] = {
            "personality_metric_datapoints": sum(
                int(row["n_datasets"])
                for row in long_rows
                if row["model_pair"] == pair_label and row["personality"] != "Overall"
            ),
            "overall_shift_datapoints": len(overall_shift_a),
            "overall_f1_datapoints": len(overall_f1_a),
        }

    mean_row: dict[str, str] = {"model_pair": "Mean pairwise"}
    pair_labels = [MODEL_PAIR_LABELS[pair] for pair in MODEL_PAIR_ORDER]
    for col in WIDE_COLUMNS:
        values = []
        for pair_label in pair_labels:
            value = wide_rows_by_pair[pair_label][col]
            parsed = parse_float(value)
            if not math.isnan(parsed):
                values.append(parsed)
        if values:
            mean_row[col] = fmt_float(mean(values))
            if len(values) < len(pair_labels):
                warnings.append(f"WARNING Mean pairwise {col}: averaged {len(values)} defined kappas, not all 3.")
        else:
            mean_row[col] = "NA"
            warnings.append(f"WARNING Mean pairwise {col}: no defined pairwise kappas.")

        personality, metric = col.rsplit(" ", 1)
        long_rows.append(
            {
                "model_pair": "Mean pairwise",
                "personality": personality,
                "metric": metric,
                "kappa": parse_float(mean_row[col]),
                "n_datasets": 48 if personality == "Overall" else 6,
                "labels_a": "",
                "labels_b": "",
                "datasets": "",
            }
        )

    wide_rows = [wide_rows_by_pair[label] for label in pair_labels]
    wide_rows.append(mean_row)
    return wide_rows, normalize_long_rows(long_rows), datapoints_by_pair


def normalize_long_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "model_pair": str(row["model_pair"]),
                "personality": str(row["personality"]),
                "metric": str(row["metric"]),
                "kappa": fmt_float_raw(float(row["kappa"])) if not isinstance(row["kappa"], str) else row["kappa"],
                "n_datasets": str(row["n_datasets"]),
                "labels_a": str(row["labels_a"]),
                "labels_b": str(row["labels_b"]),
                "datasets": str(row["datasets"]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def latex_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    colspec = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{LATEX_CAPTION}}}",
        LATEX_LABEL,
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(row.get(col, ""))) for col in columns) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary_log(
    path: Path,
    selected_paths: list[Path],
    skipped_paths: list[Path],
    model_families: list[str],
    personalities: list[str],
    datasets: list[str],
    datapoints_by_pair: dict[str, dict[str, int]],
    warnings: list[str],
    epsilon_shift: float,
    epsilon_f1: float,
) -> None:
    lines: list[str] = []
    lines.append("Cross-model kappa consistency run summary")
    lines.append("")
    lines.append(f"epsilon_shift: {epsilon_shift}")
    lines.append(f"epsilon_f1: {epsilon_f1}")
    lines.append("")
    lines.append("Input files used:")
    for path_item in selected_paths:
        lines.append(f"- {path_item.as_posix()}")
    lines.append("")
    lines.append("Skipped sentiment summary files outside requested 8 personality variants:")
    for path_item in skipped_paths:
        if path_item.as_posix().endswith("/summaries/sentiment.csv"):
            lines.append(f"- {path_item.as_posix()}")
    lines.append("")
    lines.append(f"Model families: {', '.join(model_families)}")
    lines.append(f"Personality variants: {', '.join(personalities)}")
    lines.append(f"Datasets: {', '.join(datasets)}")
    lines.append("")
    lines.append("Pairwise datapoints:")
    for pair_label, info in datapoints_by_pair.items():
        lines.append(
            f"- {pair_label}: Overall shift={info['overall_shift_datapoints']}, "
            f"Overall F1={info['overall_f1_datapoints']}, "
            f"personality metric rows total={info['personality_metric_datapoints']}"
        )
    lines.append("")
    lines.append("Warnings:")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_run_summary(
    selected_paths: list[Path],
    model_families: list[str],
    personalities: list[str],
    datasets: list[str],
    datapoints_by_pair: dict[str, dict[str, int]],
    output_paths: dict[str, Path],
    warnings: list[str],
) -> None:
    print("Input files used:")
    for path in selected_paths:
        print(f"  - {path.as_posix()}")
    print(f"Model families: {', '.join(model_families)}")
    print(f"Personality variants: {', '.join(personalities)}")
    print(f"Datasets: {', '.join(datasets)}")
    print("Pairwise comparison datapoints:")
    for pair_label, info in datapoints_by_pair.items():
        print(
            f"  - {pair_label}: Overall shift={info['overall_shift_datapoints']}, "
            f"Overall F1={info['overall_f1_datapoints']}; "
            "each personality uses common datasets reported in long format"
        )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: None")
    print("Output files:")
    for path in output_paths.values():
        print(f"  - {path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-model Cohen's kappa consistency for personality-induced shift/F1 directions."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kappa_consistency"),
        help="Directory for output tables.",
    )
    parser.add_argument("--epsilon-shift", type=float, default=1e-6)
    parser.add_argument("--epsilon-f1", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, selected_paths, skipped_paths, warnings = read_summary_rows(root)
    if not selected_paths:
        raise FileNotFoundError("No requested *_newlayout/*/summaries/sentiment.csv files found.")

    discovered_families = sorted(
        {key[0] for key in summary_rows},
        key=lambda family: MODEL_FAMILY_ORDER.index(family) if family in MODEL_FAMILY_ORDER else 10**9,
    )
    discovered_personalities = sorted(
        {key[3] for key in summary_rows if key[3] in PERSONALITY_ORDER},
        key=lambda personality: PERSONALITY_ORDER.index(personality),
    )
    discovered_datasets = sorted({key[2] for key in summary_rows}, key=sort_dataset_key)

    for family in MODEL_FAMILY_ORDER:
        if family not in discovered_families:
            warnings.append(f"WARNING missing expected model family: {family}")
    for personality in PERSONALITY_ORDER:
        if personality not in discovered_personalities:
            warnings.append(f"WARNING missing expected personality variant: {personality}")

    direction_rows, debug_rows, direction_warnings = build_direction_rows(
        summary_rows=summary_rows,
        datasets=discovered_datasets,
        epsilon_shift=args.epsilon_shift,
        epsilon_f1=args.epsilon_f1,
    )
    warnings.extend(direction_warnings)

    wide_rows, long_rows, datapoints_by_pair = compute_kappas(direction_rows, warnings)

    wide_fieldnames = ["model_pair", *WIDE_COLUMNS]
    long_fieldnames = ["model_pair", "personality", "metric", "kappa", "n_datasets", "labels_a", "labels_b", "datasets"]
    debug_fieldnames = [
        "model_family",
        "model_root",
        "personality",
        "pair",
        "dataset",
        "run",
        "source_file",
        "base_shift",
        "tuned_shift",
        "delta_shift",
        "shift_dir",
        "base_f1",
        "tuned_f1",
        "delta_f1",
        "f1_dir",
        "is_complete",
        "warning",
    ]

    output_paths = {
        "wide_csv": output_dir / "kappa_consistency_table.csv",
        "markdown": output_dir / "kappa_consistency_table.md",
        "latex": output_dir / "kappa_consistency_table.tex",
        "long_csv": output_dir / "kappa_consistency_long_format.csv",
        "debug_csv": output_dir / "kappa_consistency_debug_data.csv",
        "summary_log": output_dir / "kappa_consistency_run_summary.txt",
    }

    write_csv(output_paths["wide_csv"], wide_rows, wide_fieldnames)
    output_paths["markdown"].write_text(markdown_table(wide_rows, wide_fieldnames), encoding="utf-8")
    output_paths["latex"].write_text(latex_table(wide_rows, wide_fieldnames), encoding="utf-8")
    write_csv(output_paths["long_csv"], long_rows, long_fieldnames)
    write_csv(output_paths["debug_csv"], debug_rows, debug_fieldnames)
    write_summary_log(
        path=output_paths["summary_log"],
        selected_paths=selected_paths,
        skipped_paths=skipped_paths,
        model_families=discovered_families,
        personalities=discovered_personalities,
        datasets=discovered_datasets,
        datapoints_by_pair=datapoints_by_pair,
        warnings=warnings,
        epsilon_shift=args.epsilon_shift,
        epsilon_f1=args.epsilon_f1,
    )

    print_run_summary(
        selected_paths=selected_paths,
        model_families=discovered_families,
        personalities=discovered_personalities,
        datasets=discovered_datasets,
        datapoints_by_pair=datapoints_by_pair,
        output_paths=output_paths,
        warnings=warnings,
    )


if __name__ == "__main__":
    main()
