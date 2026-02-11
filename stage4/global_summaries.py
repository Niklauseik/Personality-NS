# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


def _clean_fieldname(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [_clean_fieldname(n) for n in reader.fieldnames or []]
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {_clean_fieldname(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            cleaned["model_root"] = csv_path.parts[0] if csv_path.parts else ""
            cleaned["source_file"] = csv_path.as_posix()
            rows.append(cleaned)
        return rows


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_significance_rows(root: Path) -> list[dict[str, str]]:
    paths = list(root.glob("*_newlayout/*/summaries/sentiment_significance.csv"))
    if not paths:
        raise FileNotFoundError("No sentiment_significance.csv files found under *_newlayout/*/summaries/")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    return rows


def write_global_csvs(root: Path, output_dir: Path) -> tuple[Path, Path, int, int]:
    rows = collect_significance_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "significance_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model_root", ""),
            row.get("pair", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("test", ""),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, str | float | int]] = []
    for (model_root, pair, model, dataset, test), group_rows in sorted(grouped.items()):
        p_values = [p for p in (_to_float(r.get("p_value")) for r in group_rows) if p is not None]
        cramers_v = [v for v in (_to_float(r.get("effect_cramers_v")) for r in group_rows) if v is not None]
        effect_tv = [v for v in (_to_float(r.get("effect_tv")) for r in group_rows) if v is not None]
        effect_js = [v for v in (_to_float(r.get("effect_js")) for r in group_rows) if v is not None]
        sig_count = sum(1 for p in p_values if p < 0.05)
        p_min = min(p_values) if p_values else math.nan
        p_max = max(p_values) if p_values else math.nan
        p_med = median(p_values) if p_values else math.nan
        summary_rows.append(
            {
                "model_root": model_root,
                "pair": pair,
                "model": model,
                "dataset": dataset,
                "test": test,
                "n_rows": len(group_rows),
                "n_sig_p_lt_0.05": sig_count,
                "sig_rate": (sig_count / len(group_rows)) if group_rows else math.nan,
                "p_min": p_min,
                "p_median": p_med,
                "p_max": p_max,
                "effect_cramers_v_median": median(cramers_v) if cramers_v else math.nan,
                "effect_tv_median": median(effect_tv) if effect_tv else math.nan,
                "effect_js_median": median(effect_js) if effect_js else math.nan,
            }
        )

    summary_path = output_dir / "significance_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_root",
            "pair",
            "model",
            "dataset",
            "test",
            "n_rows",
            "n_sig_p_lt_0.05",
            "sig_rate",
            "p_min",
            "p_median",
            "p_max",
            "effect_cramers_v_median",
            "effect_tv_median",
            "effect_js_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return long_path, summary_path, len(rows), len(summary_rows)


def _median(values: list[float]) -> float:
    return median(values) if values else math.nan


def _build_heatmap(rows: list[dict[str, str]], out_path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("pair", ""), row.get("model", ""), row.get("dataset", ""))
        grouped[key].append(row)

    row_keys = sorted({(pair, model) for pair, model, _ in grouped.keys()})
    col_keys = sorted({dataset for _, _, dataset in grouped.keys()})
    if not row_keys or not col_keys:
        return

    matrix = np.zeros((len(row_keys), len(col_keys)))
    for i, (pair, model) in enumerate(row_keys):
        for j, dataset in enumerate(col_keys):
            group = grouped.get((pair, model, dataset), [])
            p_values = [p for p in (_to_float(r.get("p_value")) for r in group) if p is not None]
            if not p_values:
                matrix[i, j] = math.nan
                continue
            p_med = max(_median(p_values), 1e-300)
            matrix[i, j] = -math.log10(p_med)

    fig, ax = plt.subplots(figsize=(0.6 * len(col_keys) + 4, 0.5 * len(row_keys) + 3))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{pair}/{model}" for pair, model in row_keys])
    ax.set_title("Median -log10(p-value) by pair/model and dataset")
    fig.colorbar(im, ax=ax, label="-log10(p-value)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_effect_plot(rows: list[dict[str, str]], out_path: Path, metric: str) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _to_float(row.get(metric))
        if value is None:
            continue
        key = (row.get("pair", ""), row.get("model", ""))
        grouped[key].append(value)

    keys = sorted(grouped.keys())
    data = [grouped[key] for key in keys]
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(0.6 * len(keys) + 4, 4))
    ax.boxplot(data, vert=True, labels=[f"{pair}/{model}" for pair, model in keys], showfliers=False)
    ax.set_title(f"Effect size distribution ({metric})")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_global_plots(
    root: Path,
    output_dir: Path,
    effect_metric: str = "effect_cramers_v",
) -> list[tuple[str, Path, Path]]:
    rows = collect_significance_rows(root)
    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_root[row.get("model_root", "")].append(row)

    written: list[tuple[str, Path, Path]] = []
    for model_root, model_rows in sorted(by_root.items()):
        model_dir = output_dir / "plots" / (model_root or "unknown_model_root")
        heatmap_path = model_dir / "significance_heatmap.png"
        effect_path = model_dir / f"effect_{effect_metric}.png"
        _build_heatmap(model_rows, heatmap_path)
        _build_effect_plot(model_rows, effect_path, effect_metric)
        written.append((model_root, heatmap_path, effect_path))
    return written

