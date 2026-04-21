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

PAIR_TITLES = {
    "decision": "Decision (F/T)",
    "energy": "Energy (E/I)",
    "information": "Information (N/S)",
    "execution": "Execution (J/P)",
}

PAIR_COLORS = {
    "decision": ("#d95f02", "#1b9e77"),
    "energy": ("#c44e52", "#4c72b0"),
    "information": ("#dd8452", "#55a868"),
    "execution": ("#8172b3", "#64b5cd"),
}

MODEL_TITLES = {
    "llama-3b_newlayout": "Llama-3.2-3B",
    "qwen-3b_newlayout": "Qwen2.5-3B",
    "qwen-7b_newlayout": "Qwen2.5-7B",
}

DATASET_ORDER = ["fiqasa", "imdb", "imdb_sklearn", "mental", "news", "sst2"]
DATASET_LABELS = {
    "fiqasa": "FiQA-SA",
    "imdb": "IMDb",
    "imdb_sklearn": "IMDb-Sklearn",
    "mental": "Mental",
    "news": "News",
    "sst2": "SST-2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3-panel delta-F1 bar charts for one MBTI pair.")
    parser.add_argument(
        "--pair",
        default="decision",
        choices=sorted(PAIR_MODELS),
        help="MBTI dimension pair to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for the exported figure files.",
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


def _build_panel_plot(pair: str, output_dir: Path) -> list[Path]:
    left_model, right_model = PAIR_MODELS[pair]
    left_color, right_color = PAIR_COLORS[pair]
    df = _load_pair_frame(pair)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )

    model_order = list(MODEL_TITLES)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    width = 0.34
    x_positions = list(range(len(DATASET_ORDER)))
    max_abs_delta = df["delta_f1_macro_strict"].abs().max()
    y_limit = max(0.08, max_abs_delta * 1.18)

    for idx, model_root in enumerate(model_order):
        ax = axes[idx]
        sub = df[df["model_root"] == model_root].set_index(["dataset", "model"])
        left_values = [sub.loc[(dataset, left_model), "delta_f1_macro_strict"] for dataset in DATASET_ORDER]
        right_values = [sub.loc[(dataset, right_model), "delta_f1_macro_strict"] for dataset in DATASET_ORDER]

        ax.bar(
            [x - width / 2 for x in x_positions],
            left_values,
            width=width,
            color=left_color,
            edgecolor="#222222",
            linewidth=0.8,
            label=left_model,
            zorder=3,
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            right_values,
            width=width,
            color=right_color,
            edgecolor="#222222",
            linewidth=0.8,
            label=right_model,
            zorder=3,
        )

        ax.axhline(0, color="#222222", linewidth=1.0, zorder=2)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=1)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([DATASET_LABELS[name] for name in DATASET_ORDER], rotation=25, ha="right")
        ax.set_title(MODEL_TITLES[model_root], pad=10)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Δ Macro-F1 vs. base")
    fig.suptitle(f"{PAIR_TITLES[pair]}: dataset-level Δ Macro-F1 across model families", y=1.02, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{pair}_delta_f1_panels"
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    args = _parse_args()
    output_paths = _build_panel_plot(args.pair, Path(args.output_dir))
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
