from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_CSV = REPO_ROOT / "global_performance" / "performance_long.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

PAIR_MODELS = {
    "decision": ("F", "T"),
    "energy": ("E", "I"),
    "information": ("N", "S"),
    "execution": ("J", "P"),
}

MODEL_ORDER = ["llama-3b_newlayout", "qwen-3b_newlayout", "qwen-7b_newlayout"]
MODEL_TITLES = {
    "llama-3b_newlayout": "Llama-3.2-3B",
    "qwen-3b_newlayout": "Qwen2.5-3B",
    "qwen-7b_newlayout": "Qwen2.5-7B",
}

DATASET_ORDER = ["fiqasa", "news", "mental", "imdb", "imdb_sklearn", "sst2"]
DATASET_LABELS = {
    "fiqasa": "FiQA-SA",
    "imdb": "IMDb",
    "imdb_sklearn": "IMDb-Sklearn",
    "mental": "Mental",
    "news": "News",
    "sst2": "SST-2",
}

# Three reds for the left pole, three blues for the right pole.
SERIES_COLORS = {
    "llama-3b_newlayout_left": "#b22222",
    "qwen-3b_newlayout_left": "#d95f5f",
    "qwen-7b_newlayout_left": "#f08a7c",
    "llama-3b_newlayout_right": "#1f4e8c",
    "qwen-3b_newlayout_right": "#4f81bd",
    "qwen-7b_newlayout_right": "#8fb6e8",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a compact grouped delta-F1 figure for one MBTI pair.")
    parser.add_argument(
        "--pair",
        default="decision",
        choices=sorted(PAIR_MODELS),
        help="MBTI dimension pair to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for exported figure files.",
    )
    return parser.parse_args()


def _load_pair_frame(pair: str) -> pd.DataFrame:
    left_model, right_model = PAIR_MODELS[pair]
    df = pd.read_csv(PERFORMANCE_CSV)
    df = df[(df["run"] == "avg") & (df["pair"] == pair)].copy()
    df = df[df["model"].isin([left_model, right_model])]
    if df.empty:
        raise ValueError(f"No rows found for pair={pair!r} in {PERFORMANCE_CSV}")
    return df


def _build_compact_plot(pair: str, output_dir: Path) -> list[Path]:
    left_model, right_model = PAIR_MODELS[pair]
    df = _load_pair_frame(pair).set_index(["model_root", "dataset", "model"])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    group_step = 0.84
    group_centers = [idx * group_step for idx in range(len(DATASET_ORDER))]
    width = 0.12
    offsets = [-2.5 * width, -1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width, 2.5 * width]

    legend_handles = []
    legend_labels = []

    for model_idx, model_root in enumerate(MODEL_ORDER):
        left_offset = offsets[model_idx]
        right_offset = offsets[model_idx + 3]
        left_values = [df.loc[(model_root, dataset, left_model), "delta_f1_macro_strict"] for dataset in DATASET_ORDER]
        right_values = [df.loc[(model_root, dataset, right_model), "delta_f1_macro_strict"] for dataset in DATASET_ORDER]

        left_bars = ax.bar(
            [x + left_offset for x in group_centers],
            left_values,
            width=width,
            color=SERIES_COLORS[f"{model_root}_left"],
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
        )
        right_bars = ax.bar(
            [x + right_offset for x in group_centers],
            right_values,
            width=width,
            color=SERIES_COLORS[f"{model_root}_right"],
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
        )

        legend_handles.extend([left_bars[0], right_bars[0]])
        legend_labels.extend(
            [
                f"{MODEL_TITLES[model_root]} {left_model}",
                f"{MODEL_TITLES[model_root]} {right_model}",
            ]
        )

    max_abs_delta = max(abs(v) for v in df["delta_f1_macro_strict"].tolist())
    y_limit = max(0.08, max_abs_delta * 1.18)

    ax.axhline(0, color="#222222", linewidth=1.0, zorder=2)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=1)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlim(group_centers[0] - 0.5, group_centers[-1] + 0.5)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS[name] for name in DATASET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Δ Macro-F1 vs. base")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.5,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{pair}_delta_f1_compact"
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    args = _parse_args()
    output_paths = _build_compact_plot(args.pair, Path(args.output_dir))
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
