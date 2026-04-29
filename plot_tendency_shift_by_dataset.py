# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


DEFAULT_INPUT = Path("tendency_shift_all_models.csv")
DEFAULT_OUTPUT_DIR = Path("tendency_shift_figures")

MODEL_ORDER = ["Llama-3.2-3B", "Qwen2.5-3B", "Qwen2.5-7B"]
MODEL_ALIASES = {
    "llama-3.2-3b": "Llama-3.2-3B",
    "llama-3b": "Llama-3.2-3B",
    "llama-3b_newlayout": "Llama-3.2-3B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "qwen-3b": "Qwen2.5-3B",
    "qwen-3b_newlayout": "Qwen2.5-3B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen-7b": "Qwen2.5-7B",
    "qwen-7b_newlayout": "Qwen2.5-7B",
}
MODEL_ROW_PREFIX = {
    "Llama-3.2-3B": "Llama",
    "Qwen2.5-3B": "Qwen-3B",
    "Qwen2.5-7B": "Qwen-7B",
}

VARIANT_ORDER = ["Base", "E", "I", "S", "N", "F", "T", "J", "P"]
VARIANT_ALIASES = {
    "base": "Base",
    "baseline": "Base",
    "original": "Base",
    "base model": "Base",
    "original base model": "Base",
    "original-base": "Base",
    "BASE": "Base",
}

DATASET_ORDER = ["FiQA-SA", "IMDb", "IMDb-Sklearn", "SST-2", "News", "Mental"]
DATASET_ALIASES = {
    "fiqa-sa": "FiQA-SA",
    "fiqa_sa": "FiQA-SA",
    "fiqasa": "FiQA-SA",
    "fiqasa_sentiment": "FiQA-SA",
    "imdb": "IMDb",
    "imdb_sentiment": "IMDb",
    "imdb-sklearn": "IMDb-Sklearn",
    "imdb_sklearn": "IMDb-Sklearn",
    "sst-2": "SST-2",
    "sst2": "SST-2",
    "news": "News",
    "news_sentiment": "News",
    "mental": "Mental",
    "mental_sentiment": "Mental",
}

DATASET_LABEL_ORDER = {
    "FiQA-SA": ["positive", "negative"],
    "IMDb": ["positive", "negative"],
    "IMDb-Sklearn": ["positive", "negative"],
    "SST-2": ["positive", "negative"],
    "News": ["bullish", "neutral", "bearish"],
    "Mental": ["normal", "depression"],
}
LABEL_DISPLAY = {
    "positive": "Positive",
    "negative": "Negative",
    "bullish": "Bullish",
    "neutral": "Neutral",
    "bearish": "Bearish",
    "normal": "Normal",
    "depression": "Depression",
}
LABEL_COLORS = {
    "positive": "#4C78A8",
    "negative": "#D65F5F",
    "bullish": "#59A14F",
    "neutral": "#BAB0AC",
    "bearish": "#D65F5F",
    "normal": "#4C78A8",
    "depression": "#B07AA1",
}

REQUIRED_COLUMNS = {"model", "dataset", "variant", "label", "proportion", "macro_f1"}
PROPORTION_TOL = 1e-3


@dataclass(frozen=True)
class PlotRow:
    model: str
    variant: str
    row_label: str
    proportions: dict[str, float]
    macro_f1: float | None


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot downstream tendency shift stacked bars by subjective dataset."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV with columns: model,dataset,variant,label,proportion,macro_f1.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PDF and PNG files will be saved.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG/PDF export DPI.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def canonical_text(value: object) -> str:
    return str(value).strip()


def normalize_key(value: object) -> str:
    return canonical_text(value).lower().replace(" ", "_")


def canonical_model(value: object) -> str:
    text = canonical_text(value)
    return MODEL_ALIASES.get(normalize_key(text), text)


def canonical_dataset(value: object) -> str:
    text = canonical_text(value)
    return DATASET_ALIASES.get(normalize_key(text), text)


def canonical_variant(value: object) -> str:
    text = canonical_text(value)
    if text in VARIANT_ALIASES:
        return VARIANT_ALIASES[text]
    lowered = text.lower()
    if lowered in VARIANT_ALIASES:
        return VARIANT_ALIASES[lowered]
    upper = text.upper()
    if upper in {"E", "I", "S", "N", "F", "T", "J", "P"}:
        return upper
    return text


def canonical_label(value: object) -> str:
    return canonical_text(value).lower()


def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={col: str(col).lstrip("\ufeff").strip() for col in df.columns})
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["model"] = df["model"].map(canonical_model)
    df["dataset"] = df["dataset"].map(canonical_dataset)
    df["variant"] = df["variant"].map(canonical_variant)
    df["label"] = df["label"].map(canonical_label)
    df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")

    bad_prop = df["proportion"].isna()
    if bad_prop.any():
        warn(f"{int(bad_prop.sum())} rows have non-numeric proportion values; treating them as 0.")
        df.loc[bad_prop, "proportion"] = 0.0

    out_of_range = (df["proportion"] < 0.0) | (df["proportion"] > 1.0)
    if out_of_range.any():
        warn(f"{int(out_of_range.sum())} rows have proportions outside [0, 1]; clipping before plotting.")
        df["proportion"] = df["proportion"].clip(lower=0.0, upper=1.0)

    bad_f1 = df["macro_f1"].notna() & ((df["macro_f1"] < 0.0) | (df["macro_f1"] > 1.0))
    if bad_f1.any():
        warn(f"{int(bad_f1.sum())} rows have Macro-F1 outside [0, 1]; those displayed values will be blank.")
        df.loc[bad_f1, "macro_f1"] = math.nan

    unknown_datasets = sorted(set(df["dataset"]) - set(DATASET_ORDER))
    if unknown_datasets:
        warn(f"Unknown datasets found and ignored by the six-figure loop: {unknown_datasets}")
    unknown_models = sorted(set(df["model"]) - set(MODEL_ORDER))
    if unknown_models:
        warn(f"Unknown model families found and ignored by the ordered rows: {unknown_models}")
    unknown_variants = sorted(set(df["variant"]) - set(VARIANT_ORDER))
    if unknown_variants:
        warn(f"Unknown variants found and ignored by the ordered rows: {unknown_variants}")

    return df


def row_display_label(model: str, variant: str) -> str:
    return f"{MODEL_ROW_PREFIX.get(model, model)} {variant}"


def macro_f1_for_group(group: pd.DataFrame, dataset: str, model: str, variant: str) -> float | None:
    values = group["macro_f1"].dropna().astype(float).unique().tolist()
    if not values:
        warn(f"{dataset}: missing Macro-F1 for {model} {variant}.")
        return None
    rounded = sorted({round(value, 10) for value in values})
    if len(rounded) > 1:
        warn(f"{dataset}: multiple Macro-F1 values for {model} {variant}; using the first value.")
    return float(values[0])


def proportions_for_group(
    group: pd.DataFrame,
    dataset: str,
    model: str,
    variant: str,
    label_order: list[str],
) -> dict[str, float]:
    duplicated = group.duplicated(subset=["label"], keep=False)
    if duplicated.any():
        warn(f"{dataset}: duplicate label rows for {model} {variant}; summing duplicate proportions.")

    label_sums = group.groupby("label", dropna=False)["proportion"].sum().to_dict()
    unexpected = sorted(set(label_sums) - set(label_order))
    if unexpected:
        warn(f"{dataset}: unexpected labels for {model} {variant} ignored in plot: {unexpected}.")

    missing = [label for label in label_order if label not in label_sums]
    if missing:
        warn(f"{dataset}: missing labels for {model} {variant}; filling with 0: {missing}.")

    proportions = {label: float(label_sums.get(label, 0.0)) for label in label_order}
    total = sum(proportions.values())
    if total <= 0.0:
        warn(f"{dataset}: non-positive total proportion for {model} {variant}; row will be shown as zeros.")
        return proportions

    if abs(total - 1.0) > PROPORTION_TOL:
        warn(f"{dataset}: proportions for {model} {variant} sum to {total:.6f}; normalising to 1.")
        proportions = {label: value / total for label, value in proportions.items()}

    return proportions


def build_plot_rows(df: pd.DataFrame, dataset: str) -> tuple[list[PlotRow], list[float]]:
    dataset_df = df.loc[df["dataset"] == dataset].copy()
    label_order = DATASET_LABEL_ORDER[dataset]
    rows: list[PlotRow] = []
    separators: list[float] = []

    for model_index, model in enumerate(MODEL_ORDER):
        block_start = len(rows)
        for variant in VARIANT_ORDER:
            group = dataset_df.loc[
                (dataset_df["model"] == model) & (dataset_df["variant"] == variant)
            ].copy()
            if group.empty:
                warn(f"{dataset}: missing row for {model} {variant}; skipping.")
                continue
            rows.append(
                PlotRow(
                    model=model,
                    variant=variant,
                    row_label=row_display_label(model, variant),
                    proportions=proportions_for_group(group, dataset, model, variant, label_order),
                    macro_f1=macro_f1_for_group(group, dataset, model, variant),
                )
            )

        if model_index < len(MODEL_ORDER) - 1 and len(rows) > block_start:
            separators.append(len(rows) - 0.5)

    return rows, separators


def annotate_label_directions(ax: plt.Axes, dataset: str) -> None:
    label_order = DATASET_LABEL_ORDER[dataset]
    if len(label_order) == 2:
        positions = [0.0, 1.0]
        aligns = ["left", "right"]
    else:
        positions = [0.0, 0.5, 1.0]
        aligns = ["left", "center", "right"]

    for label, x_pos, align in zip(label_order, positions, aligns, strict=True):
        ax.text(
            x_pos,
            1.018,
            LABEL_DISPLAY.get(label, label.title()),
            transform=ax.get_xaxis_transform(),
            ha=align,
            va="bottom",
            fontsize=7.5,
            fontweight="semibold",
            color="#333333",
            clip_on=False,
        )


def draw_tendency_figure(
    dataset: str,
    rows: list[PlotRow],
    separators: list[float],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    if not rows:
        warn(f"{dataset}: no plottable rows; no figure written.")
        return []

    n_rows = len(rows)
    fig_height = max(4.2, 1.45 + 0.31 * n_rows)
    fig, (ax, f1_ax) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(8.0, fig_height),
        gridspec_kw={"width_ratios": [5.8, 0.95], "wspace": 0.04},
    )

    label_order = DATASET_LABEL_ORDER[dataset]
    y_positions = list(range(n_rows))

    for y_pos, row in zip(y_positions, rows, strict=True):
        left = 0.0
        for label in label_order:
            width = row.proportions.get(label, 0.0)
            if width <= 0.0:
                continue
            ax.barh(
                y_pos,
                width,
                left=left,
                height=0.68,
                color=LABEL_COLORS[label],
                edgecolor="white",
                linewidth=0.5,
            )
            left += width

        f1_text = "" if row.macro_f1 is None or math.isnan(row.macro_f1) else f"{row.macro_f1:.3f}"
        f1_ax.text(0.5, y_pos, f1_text, ha="center", va="center", fontsize=7.2, color="#222222")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row.row_label for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Prediction proportion")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=3, color="#777777")

    for separator in separators:
        ax.axhline(separator, color="#D1D5DB", linewidth=0.7)
        f1_ax.axhline(separator, color="#D1D5DB", linewidth=0.7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")

    f1_ax.set_xlim(0.0, 1.0)
    f1_ax.set_xticks([])
    f1_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in ["top", "right", "bottom"]:
        f1_ax.spines[spine].set_visible(False)
    f1_ax.spines["left"].set_color("#D1D5DB")
    f1_ax.spines["left"].set_linewidth(0.7)
    f1_ax.text(
        0.5,
        1.018,
        "Macro-F1",
        transform=f1_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="semibold",
        color="#333333",
        clip_on=False,
    )

    annotate_label_directions(ax, dataset)
    fig.suptitle(dataset, y=0.992, fontsize=10.5, fontweight="semibold")
    fig.subplots_adjust(left=0.23, right=0.965, top=0.90, bottom=0.075)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"fig_tendency_{dataset}.png"
    pdf_path = output_dir / f"fig_tendency_{dataset}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def generate_all_figures(input_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    df = read_input_csv(input_path)
    written: list[Path] = []
    for dataset in DATASET_ORDER:
        rows, separators = build_plot_rows(df, dataset)
        written.extend(draw_tendency_figure(dataset, rows, separators, output_dir, dpi))
    return written


def main() -> None:
    args = parse_args()
    written = generate_all_figures(Path(args.input), Path(args.output_dir), args.dpi)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
