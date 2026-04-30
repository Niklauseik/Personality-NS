# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import to_hex, to_rgb
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

VARIANT_ORDER = ["Base", "F", "T", "E", "I", "S", "N", "J", "P"]
VARIANT_ALIASES = {
    "base": "Base",
    "baseline": "Base",
    "original": "Base",
    "base_model": "Base",
    "original_base_model": "Base",
    "original-base": "Base",
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

# Edit this mapping when a dataset needs a specific class order.
# For datasets not listed here, labels are taken from their first appearance in the CSV.
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

BASE_COLUMNS = {"model", "dataset", "variant", "label", "proportion"}
F1_COLUMN_CANDIDATES = ("f1", "macro_f1", "macro-f1", "macro_f1_score")
PROPORTION_TOL = 1e-3

BAR_HEIGHT = 0.66
BASE_HATCH = "////"
ROW_BAND_COLOR = "#F8FAFC"
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#374151"


@dataclass(frozen=True)
class PlotRow:
    variant: str
    proportions: dict[str, float]
    f1: float | None


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one publication-style tendency-shift figure per dataset, "
            "with three aligned model-family panels."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Input CSV with model,dataset,variant,label,proportion and either "
            "f1 or macro_f1 columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PDF and PNG files will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to plot. Defaults to all datasets in the CSV.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster export DPI.")
    parser.add_argument(
        "--width",
        type=float,
        default=10.8,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=3.45,
        help="Figure height in inches.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "figure.titlesize": 10.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "hatch.linewidth": 0.45,
        }
    )


def canonical_text(value: object) -> str:
    return str(value).strip()


def normalize_key(value: object) -> str:
    text = canonical_text(value).lower()
    return text.replace(" ", "_")


def canonical_model(value: object) -> str:
    text = canonical_text(value)
    return MODEL_ALIASES.get(normalize_key(text), text)


def canonical_dataset(value: object) -> str:
    text = canonical_text(value)
    return DATASET_ALIASES.get(normalize_key(text), text)


def canonical_variant(value: object) -> str:
    text = canonical_text(value)
    key = normalize_key(text)
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]

    upper = text.upper()
    if upper in {"F", "T", "E", "I", "S", "N", "J", "P"}:
        return upper

    for variant in ("F", "T", "E", "I", "S", "N", "J", "P"):
        if text.startswith(variant):
            return variant
    return text


def canonical_label(value: object) -> str:
    return canonical_text(value).lower()


def display_label(label: str) -> str:
    return LABEL_DISPLAY.get(label, label.replace("_", " ").title())


def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={col: str(col).lstrip("\ufeff").strip() for col in df.columns})

    lower_to_original = {str(col).lower(): col for col in df.columns}
    f1_source = next((lower_to_original[col] for col in F1_COLUMN_CANDIDATES if col in lower_to_original), None)
    if f1_source is None:
        raise ValueError(
            "Input CSV is missing an F1 column. Expected one of: "
            f"{', '.join(F1_COLUMN_CANDIDATES)}."
        )
    if f1_source != "f1":
        df = df.rename(columns={f1_source: "f1"})

    missing = BASE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["model"] = df["model"].map(canonical_model)
    df["dataset"] = df["dataset"].map(canonical_dataset)
    df["variant"] = df["variant"].map(canonical_variant)
    df["label"] = df["label"].map(canonical_label)
    df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    bad_prop = df["proportion"].isna()
    if bad_prop.any():
        warn(f"{int(bad_prop.sum())} rows have non-numeric proportions; treating them as 0.")
        df.loc[bad_prop, "proportion"] = 0.0

    out_of_range = (df["proportion"] < 0.0) | (df["proportion"] > 1.0)
    if out_of_range.any():
        warn(f"{int(out_of_range.sum())} rows have proportions outside [0, 1]; clipping.")
        df["proportion"] = df["proportion"].clip(lower=0.0, upper=1.0)

    bad_f1 = df["f1"].notna() & ((df["f1"] < 0.0) | (df["f1"] > 1.0))
    if bad_f1.any():
        warn(f"{int(bad_f1.sum())} rows have F1 outside [0, 1]; those displayed values will be blank.")
        df.loc[bad_f1, "f1"] = math.nan

    unknown_models = sorted(set(df["model"]) - set(MODEL_ORDER))
    if unknown_models:
        warn(f"Unknown model families found and ignored by the three-panel layout: {unknown_models}")

    unknown_variants = sorted(set(df["variant"]) - set(VARIANT_ORDER))
    if unknown_variants:
        warn(f"Unknown variants found and ignored by the ordered panel rows: {unknown_variants}")

    return df


def datasets_to_plot(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    present = set(df["dataset"])
    if requested:
        datasets = [canonical_dataset(dataset) for dataset in requested]
    else:
        datasets = [dataset for dataset in DATASET_ORDER if dataset in present]
        datasets.extend(sorted(present.difference(DATASET_ORDER)))

    missing = [dataset for dataset in datasets if dataset not in present]
    if missing:
        warn(f"Requested datasets not found in input and skipped: {missing}")
    return [dataset for dataset in datasets if dataset in present]


def label_order_for_dataset(df: pd.DataFrame, dataset: str) -> list[str]:
    if dataset in DATASET_LABEL_ORDER:
        return DATASET_LABEL_ORDER[dataset]

    labels = df.loc[df["dataset"] == dataset, "label"].dropna().tolist()
    ordered: list[str] = []
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def f1_for_group(group: pd.DataFrame, dataset: str, model: str, variant: str) -> float | None:
    values = group["f1"].dropna().astype(float).unique().tolist()
    if not values:
        warn(f"{dataset}: missing F1 for {model} {variant}.")
        return None
    rounded = sorted({round(value, 10) for value in values})
    if len(rounded) > 1:
        warn(f"{dataset}: multiple F1 values for {model} {variant}; using the first value.")
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
        warn(f"{dataset}: duplicate labels for {model} {variant}; summing duplicate proportions.")

    label_sums = group.groupby("label", dropna=False)["proportion"].sum().to_dict()
    unexpected = sorted(set(label_sums) - set(label_order))
    if unexpected:
        warn(f"{dataset}: labels ignored for {model} {variant}: {unexpected}.")

    missing = [label for label in label_order if label not in label_sums]
    if missing:
        warn(f"{dataset}: missing labels for {model} {variant}; filling with 0: {missing}.")

    proportions = {label: float(label_sums.get(label, 0.0)) for label in label_order}
    total = sum(proportions.values())
    if total <= 0.0:
        warn(f"{dataset}: non-positive total proportion for {model} {variant}; row will be blank.")
        return proportions

    if abs(total - 1.0) > PROPORTION_TOL:
        warn(f"{dataset}: proportions for {model} {variant} sum to {total:.6f}; normalising to 1.")
        proportions = {label: value / total for label, value in proportions.items()}

    return proportions


def build_panel_rows(
    df: pd.DataFrame,
    dataset: str,
    label_order: list[str],
) -> dict[str, list[PlotRow]]:
    dataset_df = df.loc[df["dataset"] == dataset].copy()
    rows_by_model: dict[str, list[PlotRow]] = {}

    for model in MODEL_ORDER:
        model_rows: list[PlotRow] = []
        for variant in VARIANT_ORDER:
            group = dataset_df.loc[
                (dataset_df["model"] == model) & (dataset_df["variant"] == variant)
            ].copy()
            if group.empty:
                warn(f"{dataset}: missing row for {model} {variant}; drawing an empty row.")
                proportions = {label: 0.0 for label in label_order}
                f1 = None
            else:
                proportions = proportions_for_group(group, dataset, model, variant, label_order)
                f1 = f1_for_group(group, dataset, model, variant)
            model_rows.append(PlotRow(variant=variant, proportions=proportions, f1=f1))
        rows_by_model[model] = model_rows

    return rows_by_model


def choose_label_colors(label_order: list[str]) -> dict[str, str]:
    # Okabe-Ito inspired cycle: colorblind-safe, high contrast, and deterministic by class order.
    base_palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
        "#F0E442",
    ]
    if len(label_order) <= len(base_palette):
        colors = base_palette[: len(label_order)]
    else:
        cmap = mpl.colormaps["tab20"]
        colors = [to_hex(cmap(i / max(1, len(label_order) - 1))) for i in range(len(label_order))]
    return dict(zip(label_order, colors, strict=True))


def relative_luminance(color: str) -> float:
    def linearize(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def text_color_for_fill(fill_color: str) -> str:
    return "#FFFFFF" if relative_luminance(fill_color) < 0.32 else "#111827"


def label_font_size(width: float) -> float:
    if width >= 0.18:
        return 6.8
    if width >= 0.12:
        return 6.2
    return 5.5


def base_reference_boundaries(rows: list[PlotRow], label_order: list[str]) -> list[float]:
    base_row = next((row for row in rows if row.variant == "Base"), None)
    if base_row is None:
        return []

    boundaries: list[float] = []
    cumulative = 0.0
    for label in label_order[:-1]:
        cumulative += base_row.proportions.get(label, 0.0)
        if 0.0 < cumulative < 1.0:
            boundaries.append(cumulative)
    return boundaries


def format_f1(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.2f}"


def draw_row_background(ax: plt.Axes, n_rows: int) -> None:
    for row_index in range(n_rows):
        if row_index % 2 == 0:
            ax.axhspan(row_index - 0.5, row_index + 0.5, color=ROW_BAND_COLOR, zorder=0)


def draw_stacked_row(
    ax: plt.Axes,
    row: PlotRow,
    y_pos: int,
    label_order: list[str],
    label_colors: dict[str, str],
) -> None:
    left = 0.0
    total_width = 0.0

    for label in label_order:
        width = row.proportions.get(label, 0.0)
        if width <= 0.0:
            continue

        color = label_colors[label]
        ax.barh(
            y_pos,
            width,
            left=left,
            height=BAR_HEIGHT,
            color=color,
            edgecolor="white",
            linewidth=0.42,
            zorder=2,
        )
        ax.text(
            left + width / 2.0,
            y_pos,
            f"{width * 100:.1f}%",
            ha="center",
            va="center",
            color=text_color_for_fill(color),
            fontsize=label_font_size(width),
            fontweight="semibold",
            zorder=5,
            clip_on=True,
        )
        left += width
        total_width += width

    if row.variant == "Base" and total_width > 0.0:
        ax.barh(
            y_pos,
            total_width,
            left=0.0,
            height=BAR_HEIGHT,
            facecolor="none",
            edgecolor="#111827",
            linewidth=0.18,
            hatch=BASE_HATCH,
            alpha=0.38,
            zorder=3,
        )


def style_bar_axis(ax: plt.Axes, is_first_panel: bool, n_rows: int) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, zorder=1)
    ax.tick_params(axis="x", length=2.5, width=0.55, color="#6B7280", pad=2)
    ax.tick_params(axis="y", length=0, pad=4)

    ax.set_yticks(range(n_rows))
    if is_first_panel:
        ax.set_yticklabels(VARIANT_ORDER)
    else:
        ax.tick_params(axis="y", labelleft=False)

    for row_edge in [index + 0.5 for index in range(n_rows - 1)]:
        ax.axhline(row_edge, color="#EDF2F7", linewidth=0.45, zorder=1)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#6B7280")
    ax.spines["bottom"].set_linewidth(0.6)


def style_f1_axis(ax: plt.Axes, n_rows: int) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    ax.tick_params(axis="y", left=False, labelleft=False)

    for row_edge in [index + 0.5 for index in range(n_rows - 1)]:
        ax.axhline(row_edge, color="#EDF2F7", linewidth=0.45, zorder=1)

    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(0.65)


def draw_model_panel(
    bar_ax: plt.Axes,
    f1_ax: plt.Axes,
    model: str,
    rows: list[PlotRow],
    label_order: list[str],
    label_colors: dict[str, str],
    is_first_panel: bool,
) -> None:
    n_rows = len(rows)
    draw_row_background(bar_ax, n_rows)
    draw_row_background(f1_ax, n_rows)

    for y_pos, row in enumerate(rows):
        draw_stacked_row(bar_ax, row, y_pos, label_order, label_colors)
        f1_ax.text(
            0.5,
            y_pos,
            format_f1(row.f1),
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=7.1,
            fontweight="semibold",
            zorder=4,
        )

    for boundary in base_reference_boundaries(rows, label_order):
        bar_ax.axvline(
            boundary,
            color="#111827",
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.85,
            alpha=0.78,
            zorder=4,
        )

    style_bar_axis(bar_ax, is_first_panel, n_rows)
    style_f1_axis(f1_ax, n_rows)
    bar_ax.set_title(model, pad=8, fontweight="semibold", color=TEXT_COLOR)
    f1_ax.set_title("F1", pad=8, fontweight="semibold", color=TEXT_COLOR)


def draw_dataset_figure(
    dataset: str,
    rows_by_model: dict[str, list[PlotRow]],
    label_order: list[str],
    output_dir: Path,
    dpi: int,
    width: float,
    height: float,
) -> list[Path]:
    if not label_order:
        warn(f"{dataset}: no labels found; no figure written.")
        return []

    label_colors = choose_label_colors(label_order)
    n_rows = len(VARIANT_ORDER)

    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        len(MODEL_ORDER),
        left=0.06,
        right=0.988,
        top=0.755,
        bottom=0.15,
        wspace=0.045,
    )

    first_bar_ax: plt.Axes | None = None

    for model_index, model in enumerate(MODEL_ORDER):
        subgrid = outer[model_index].subgridspec(
            1,
            2,
            width_ratios=[5.8, 0.68],
            wspace=0.018,
        )
        bar_ax = fig.add_subplot(subgrid[0, 0], sharey=first_bar_ax)
        if first_bar_ax is None:
            first_bar_ax = bar_ax
        f1_ax = fig.add_subplot(subgrid[0, 1], sharey=first_bar_ax)

        draw_model_panel(
            bar_ax=bar_ax,
            f1_ax=f1_ax,
            model=model,
            rows=rows_by_model[model],
            label_order=label_order,
            label_colors=label_colors,
            is_first_panel=model_index == 0,
        )

    if first_bar_ax is not None:
        first_bar_ax.set_ylim(n_rows - 0.5, -0.5)

    legend_handles = [
        Patch(facecolor=label_colors[label], edgecolor="none", label=display_label(label))
        for label in label_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.900),
        ncol=min(len(legend_handles), 5),
        frameon=False,
        handlelength=1.55,
        handletextpad=0.45,
        columnspacing=1.0,
        labelcolor=MUTED_TEXT_COLOR,
    )

    fig.suptitle(dataset, y=0.982, fontweight="semibold", color=TEXT_COLOR)
    fig.text(
        0.5,
        0.047,
        "Prediction proportion",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED_TEXT_COLOR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "-")
    png_path = output_dir / f"fig_tendency_{safe_dataset}.png"
    pdf_path = output_dir / f"fig_tendency_{safe_dataset}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def generate_figures(
    input_path: Path,
    output_dir: Path,
    datasets: list[str] | None,
    dpi: int,
    width: float,
    height: float,
) -> list[Path]:
    configure_matplotlib()
    df = read_input_csv(input_path)
    written: list[Path] = []

    for dataset in datasets_to_plot(df, datasets):
        label_order = label_order_for_dataset(df, dataset)
        rows_by_model = build_panel_rows(df, dataset, label_order)
        written.extend(
            draw_dataset_figure(
                dataset=dataset,
                rows_by_model=rows_by_model,
                label_order=label_order,
                output_dir=output_dir,
                dpi=dpi,
                width=width,
                height=height,
            )
        )
    return written


def main() -> None:
    args = parse_args()
    written = generate_figures(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        datasets=args.datasets,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
