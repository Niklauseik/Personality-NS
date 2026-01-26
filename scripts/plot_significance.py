#!/usr/bin/env python3
"""Plot heatmap and effect-size summaries from sentiment significance CSVs."""
from __future__ import annotations

import argparse
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
            cleaned = { _clean_fieldname(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items() }
            cleaned["source_file"] = str(csv_path)
            rows.append(cleaned)
        return rows


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(root: Path) -> list[dict[str, str]]:
    paths = list(root.glob("*_newlayout/*/summaries/sentiment_significance.csv"))
    if not paths:
        raise SystemExit("No sentiment_significance.csv files found under *_newlayout/*/summaries/")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    return rows


def _median(values: list[float]) -> float:
    return median(values) if values else math.nan


def build_heatmap(rows: list[dict[str, str]], out_path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("pair", ""), row.get("model", ""), row.get("dataset", ""))
        grouped[key].append(row)

    row_keys = sorted({(pair, model) for pair, model, _ in grouped.keys()})
    col_keys = sorted({dataset for _, _, dataset in grouped.keys()})

    if not row_keys or not col_keys:
        raise SystemExit("No data for heatmap.")

    matrix = np.zeros((len(row_keys), len(col_keys)))
    for i, (pair, model) in enumerate(row_keys):
        for j, dataset in enumerate(col_keys):
            group = grouped.get((pair, model, dataset), [])
            p_values = [
                p for p in (_to_float(r.get("p_value")) for r in group) if p is not None
            ]
            if not p_values:
                matrix[i, j] = math.nan
                continue
            p_med = _median(p_values)
            p_med = max(p_med, 1e-300)
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_effect_plot(rows: list[dict[str, str]], out_path: Path, metric: str) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _to_float(row.get(metric))
        if value is None:
            continue
        key = (row.get("pair", ""), row.get("model", ""))
        grouped[key].append(value)

    keys = sorted(grouped.keys())
    data = [grouped[key] for key in keys]

    fig, ax = plt.subplots(figsize=(0.6 * len(keys) + 4, 4))
    ax.boxplot(data, vert=True, labels=[f"{pair}/{model}" for pair, model in keys],
               showfliers=False)
    ax.set_title(f"Effect size distribution ({metric})")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot significance summaries.")
    parser.add_argument("--output-dir", default="summaries", help="Directory for output images")
    parser.add_argument(
        "--effect-metric",
        default="effect_cramers_v",
        choices=["effect_cramers_v", "effect_tv", "effect_js"],
        help="Effect size column to plot",
    )
    args = parser.parse_args()

    rows = _load_rows(Path("."))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / "significance_heatmap.png"
    effect_path = out_dir / f"effect_{args.effect_metric}.png"

    build_heatmap(rows, heatmap_path)
    build_effect_plot(rows, effect_path, args.effect_metric)

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {effect_path}")


if __name__ == "__main__":
    main()
