# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import pandas as pd

try:  # pragma: no cover - optional at import time, required for p-values
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


DEFAULT_INPUT = "benchmark_deltas_by_dataset.md"
DEFAULT_OUTPUT_DIR = "analysis_outputs/benchmark_drop_ttest"
DEFAULT_P_THRESHOLD = 0.05
DEFAULT_METRIC_COLUMN = "Delta Acc"

MODEL_ORDER = ["Llama-3.2-3B", "Qwen2.5-3B", "Qwen2.5-7B"]
DATASET_ORDER = ["ARC (easy)", "BoolQ", "GSM8K"]
DATASET_DISPLAY = {
    "ARC (easy)": "ARC-Easy",
    "BoolQ": "BoolQ",
    "GSM8K": "GSM8K",
    "Overall": "Overall",
}


@dataclass(frozen=True)
class TTestResult:
    n: int
    mean_delta: float
    std_delta: float
    t_statistic: float
    p_drop: float


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    text = text.replace("+", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def _parse_markdown_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None

    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))

    if not rows:
        raise ValueError(f"No markdown table rows found in {path}")
    return pd.DataFrame(rows)


def _load_input_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark delta input not found: {input_path}")
    if input_path.suffix.lower() in {".md", ".markdown"}:
        return _parse_markdown_table(input_path)
    return pd.read_csv(input_path, dtype=str).fillna("")


def _normalise_delta_rows(input_path: Path, metric_column: str) -> pd.DataFrame:
    raw = _load_input_table(input_path)
    required = ["Model", "Dimension", "Type", "Dataset", metric_column]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        delta = _parse_float(row.get(metric_column))
        if delta is None:
            continue
        rows.append(
            {
                "model": str(row["Model"]).strip(),
                "dimension": str(row["Dimension"]).strip(),
                "type": str(row["Type"]).strip(),
                "dataset": str(row["Dataset"]).strip(),
                "metric": metric_column,
                "delta": delta,
                "delta_pp": delta * 100.0,
                "source_file": input_path.as_posix(),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable benchmark delta rows found in {input_path}")
    return df


def _ordered_present(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    seen = {str(value) for value in values}
    ordered = [value for value in preferred if value in seen]
    ordered.extend(sorted(value for value in seen if value not in ordered))
    return ordered


def _ttest_less(values: Sequence[float]) -> TTestResult:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return TTestResult(0, math.nan, math.nan, math.nan, math.nan)
    mean_delta = float(mean(clean))
    std_delta = float(stdev(clean)) if n > 1 else math.nan
    if n < 2:
        return TTestResult(n, mean_delta, std_delta, math.nan, math.nan)
    if std_delta == 0:
        if mean_delta < 0:
            return TTestResult(n, mean_delta, std_delta, -math.inf, 0.0)
        if mean_delta > 0:
            return TTestResult(n, mean_delta, std_delta, math.inf, 1.0)
        return TTestResult(n, mean_delta, std_delta, 0.0, 0.5)
    if stats is None:
        raise RuntimeError("scipy is required to compute one-sided paired t-test p-values.")
    result = stats.ttest_1samp(clean, 0.0, alternative="less")
    return TTestResult(n, mean_delta, std_delta, float(result.statistic), float(result.pvalue))


def _build_variant_overall(delta_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "dimension", "type"]
    rows: list[dict[str, object]] = []
    for keys, group in delta_df.groupby(group_cols, sort=False):
        model, dimension, model_type = keys
        rows.append(
            {
                "model": model,
                "dimension": dimension,
                "type": model_type,
                "dataset": "Overall",
                "metric": "mean_delta_across_datasets",
                "delta": float(group["delta"].mean()),
                "delta_pp": float(group["delta_pp"].mean()),
                "n_datasets": int(group["dataset"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _group_stats_rows(
    delta_df: pd.DataFrame,
    variant_overall: pd.DataFrame,
    p_threshold: float,
) -> pd.DataFrame:
    models = _ordered_present(delta_df["model"].tolist(), MODEL_ORDER)
    datasets = _ordered_present(delta_df["dataset"].tolist(), DATASET_ORDER)
    rows: list[dict[str, object]] = []

    def append_result(row_label: str, model: str, dataset: str, values: Sequence[float]) -> None:
        result = _ttest_less(values)
        rows.append(
            {
                "row_label": row_label,
                "model": model,
                "dataset": dataset,
                "n": result.n,
                "delta_mean": result.mean_delta,
                "delta_std": result.std_delta,
                "delta_mean_pp": result.mean_delta * 100.0 if math.isfinite(result.mean_delta) else math.nan,
                "delta_std_pp": result.std_delta * 100.0 if math.isfinite(result.std_delta) else math.nan,
                "t_statistic": result.t_statistic,
                "p_drop": result.p_drop,
                "significant_drop": bool(result.p_drop < p_threshold)
                if math.isfinite(result.p_drop)
                else False,
                "alternative": "mean_delta_less_than_0",
            }
        )

    for model in models:
        model_df = delta_df.loc[delta_df["model"] == model]
        for dataset in datasets:
            values = model_df.loc[model_df["dataset"] == dataset, "delta"].tolist()
            append_result("model", model, dataset, values)
        overall_values = variant_overall.loc[variant_overall["model"] == model, "delta"].tolist()
        append_result("model", model, "Overall", overall_values)

    for dataset in datasets:
        values = delta_df.loc[delta_df["dataset"] == dataset, "delta"].tolist()
        append_result("overall", "Overall", dataset, values)
    append_result("overall", "Overall", "Overall", variant_overall["delta"].tolist())

    return pd.DataFrame(rows)


def _format_delta(mean_pp: float, std_pp: float) -> str:
    if not math.isfinite(mean_pp) or not math.isfinite(std_pp):
        return "NA"
    return f"{mean_pp:+.1f} {chr(177)} {std_pp:.1f}"


def _format_p(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _build_wide_table(stats_df: pd.DataFrame, p_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_models = _ordered_present(stats_df["model"].tolist(), [*MODEL_ORDER, "Overall"])
    row_models = [model for model in row_models if model != "Overall"] + ["Overall"]

    for model in row_models:
        row: dict[str, object] = {"Model": model}
        for dataset in DATASET_ORDER:
            match = stats_df.loc[(stats_df["model"] == model) & (stats_df["dataset"] == dataset)]
            if match.empty:
                row[f"{DATASET_DISPLAY[dataset]} delta mean +/- std"] = "NA"
                row[f"{DATASET_DISPLAY[dataset]} p-drop"] = "NA"
                continue
            record = match.iloc[0]
            row[f"{DATASET_DISPLAY[dataset]} delta mean +/- std"] = _format_delta(
                float(record["delta_mean_pp"]), float(record["delta_std_pp"])
            )
            row[f"{DATASET_DISPLAY[dataset]} p-drop"] = _format_p(float(record["p_drop"]))

        overall = stats_df.loc[(stats_df["model"] == model) & (stats_df["dataset"] == "Overall")]
        if overall.empty:
            row["Overall delta mean +/- std"] = "NA"
            row["Overall p-drop"] = "NA"
            row["Significant drop?"] = "No"
        else:
            record = overall.iloc[0]
            row["Overall delta mean +/- std"] = _format_delta(
                float(record["delta_mean_pp"]), float(record["delta_std_pp"])
            )
            row["Overall p-drop"] = _format_p(float(record["p_drop"]))
            row["Significant drop?"] = "Yes" if float(record["p_drop"]) < p_threshold else "No"
        rows.append(row)

    return pd.DataFrame(rows)


def _markdown_table(wide_df: pd.DataFrame, include_overall_p: bool) -> str:
    columns = [
        "Model",
        "ARC-Easy delta mean +/- std",
        "ARC-Easy p-drop",
        "BoolQ delta mean +/- std",
        "BoolQ p-drop",
        "GSM8K delta mean +/- std",
        "GSM8K p-drop",
        "Overall delta mean +/- std",
    ]
    if include_overall_p:
        columns.append("Overall p-drop")
    columns.append("Significant drop?")

    header_labels = {
        "Model": "Model",
        "ARC-Easy delta mean +/- std": f"ARC-Easy {chr(916)} mean {chr(177)} std",
        "ARC-Easy p-drop": "p-drop",
        "BoolQ delta mean +/- std": f"BoolQ {chr(916)} mean {chr(177)} std",
        "BoolQ p-drop": "p-drop",
        "GSM8K delta mean +/- std": f"GSM8K {chr(916)} mean {chr(177)} std",
        "GSM8K p-drop": "p-drop",
        "Overall delta mean +/- std": f"Overall {chr(916)} mean {chr(177)} std",
        "Overall p-drop": "p-drop",
        "Significant drop?": "Significant drop?",
    }
    lines = [
        "| " + " | ".join(header_labels[col] for col in columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 2) + ["---"]) + " |",
    ]
    for _, row in wide_df.iterrows():
        values: list[str] = []
        for col in columns:
            value = str(row[col])
            if col == "Model" and value == "Overall":
                value = "**Overall**"
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    input_path: Path,
    output_dir: Path,
    metric_column: str = DEFAULT_METRIC_COLUMN,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    include_overall_p_in_markdown: bool = False,
) -> dict[str, Path]:
    delta_df = _normalise_delta_rows(input_path, metric_column=metric_column)
    variant_overall = _build_variant_overall(delta_df)
    stats_df = _group_stats_rows(delta_df, variant_overall, p_threshold=p_threshold)
    wide_df = _build_wide_table(stats_df, p_threshold=p_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inputs": output_dir / "benchmark_drop_ttest_inputs.csv",
        "variant_overall": output_dir / "benchmark_drop_ttest_variant_overall.csv",
        "group_stats": output_dir / "benchmark_drop_ttest_group_stats.csv",
        "table_csv": output_dir / "benchmark_drop_ttest_table.csv",
        "table_md": output_dir / "benchmark_drop_ttest_table.md",
    }
    delta_df.to_csv(paths["inputs"], index=False, encoding="utf-8-sig")
    variant_overall.to_csv(paths["variant_overall"], index=False, encoding="utf-8-sig")
    stats_df.to_csv(paths["group_stats"], index=False, encoding="utf-8-sig")
    wide_df.to_csv(paths["table_csv"], index=False, encoding="utf-8-sig")
    paths["table_md"].write_text(
        _markdown_table(wide_df, include_overall_p=include_overall_p_in_markdown),
        encoding="utf-8",
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-sided drop t-test for benchmark capability deltas."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Benchmark delta table to read (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSV/Markdown files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--metric-column",
        default=DEFAULT_METRIC_COLUMN,
        help=f"Delta column to test (default: {DEFAULT_METRIC_COLUMN}).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=DEFAULT_P_THRESHOLD,
        help=f"Significance threshold for drop decisions (default: {DEFAULT_P_THRESHOLD}).",
    )
    parser.add_argument(
        "--include-overall-p-in-markdown",
        action="store_true",
        help="Add an Overall p-drop column to the Markdown table.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    return write_outputs(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        metric_column=args.metric_column,
        p_threshold=args.p_threshold,
        include_overall_p_in_markdown=args.include_overall_p_in_markdown,
    )
