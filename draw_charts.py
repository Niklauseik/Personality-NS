# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use a font family that supports both English and CJK characters
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang HK",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from pipeline_utils import ordered_model_entries

allowed_labels = {
    "imdb_sentiment": {"negative", "positive"},
    "mental_sentiment": {"depression", "normal"},
    "news_sentiment": {"bearish", "bullish", "neutral"},
    "fiqasa_sentiment": {"negative", "positive", "neutral"},
    "imdb_sklearn": {"negative", "positive"},
    "sst2": {"negative", "positive"},
}


def collapse_extra_labels(dataset: str, model_data: dict):
    allow = allowed_labels.get(dataset, set())
    for counts in model_data.values():
        extra_total = 0
        for label in list(counts.keys()):
            if label not in allow and label != "invalid":
                extra_total += counts.pop(label)
        counts["invalid"] = counts.get("invalid", 0) + extra_total


def plot_pies(dataset: str, model_data: dict, model_order):
    fig, axs = plt.subplots(1, len(model_order), figsize=(4 * len(model_order), 5))
    fig.suptitle(f"{dataset} Prediction Distribution", fontsize=14, y=0.98)

    for idx, (model_key, model_label) in enumerate(model_order):
        ax = axs[idx] if len(model_order) > 1 else axs
        data = model_data.get(model_key, {})
        if not data:
            ax.axis("off")
            ax.set_title(model_label, pad=2)
            continue
        labels = list(data.keys())
        sizes = list(data.values())
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        ax.set_title(model_label, pad=2)

    fig.subplots_adjust(top=0.82, wspace=0.25)
    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{dataset}_prediction_distribution_pie.png", dpi=150)
    plt.close(fig)


def plot_bars(dataset: str, model_data: dict, model_order):
    all_labels = set()
    for counts in model_data.values():
        all_labels.update(counts.keys())
    labels = sorted([lbl for lbl in all_labels if lbl != "invalid"]) + (
        ["invalid"] if "invalid" in all_labels else []
    )
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.8 / max(len(model_order), 1)

    for i, (model_key, model_label) in enumerate(model_order):
        counts = [model_data.get(model_key, {}).get(lbl, 0) for lbl in labels]
        ax.bar(x + (i - (len(model_order) - 1) / 2) * width, counts, width, label=model_label)

    ax.set_title(f"{dataset} Prediction Distribution", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Count")
    ax.legend(ncol=min(len(model_order), 3), frameon=False, loc="upper right")
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.88)
    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{dataset}_prediction_distribution_bar.png", dpi=150)
    plt.close(fig)


def generate_charts(results_root: Path | str = "results", chart_data: dict | None = None):
    if chart_data is None:
        from sentiment_label_count import summarize_label_distribution

        chart_data = summarize_label_distribution(results_root)

    entries = ordered_model_entries(Path(results_root))
    if not entries:
        raise RuntimeError("No pipeline metadata found. Run stage-1 pipeline first.")
    def legend_label(entry):
        code = entry.get("code", "")
        if code == "base":
            return "BASE"
        return code.upper() or entry["display_name"]

    ordered_entries = (
        [entry for entry in entries if entry.get("role") == "base"]
        + [entry for entry in entries if entry.get("role") != "base"]
    )
    model_order = [(entry["display_name"], legend_label(entry)) for entry in ordered_entries]

    for dataset, info in chart_data.items():
        model_data = {
            model_key: dict(info["models"].get(model_key, {}))
            for model_key, _ in model_order
        }
        collapse_extra_labels(dataset, model_data)
        plot_pies(dataset, model_data, model_order)
        plot_bars(dataset, model_data, model_order)

    print("✅ 全部图表已输出到 ./plots 目录")


if __name__ == "__main__":
    generate_charts()
