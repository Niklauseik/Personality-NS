from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ft_dimension_mbti_summary.csv"
OUTPUT_PATH = ROOT / "ft_dimension_mbti_summary.png"


F_COLOR = "#F7C7D9"
T_COLOR = "#BBD7F0"
MODEL_COLOR = "#F4F6F8"
HEADER_COLOR = "#1F2937"
GRID_COLOR = "#94A3B8"
TEXT_COLOR = "#111827"


def decision_pole(mbti_type: str) -> str:
    if not isinstance(mbti_type, str) or len(mbti_type) < 3:
        return ""
    return mbti_type[2].upper()


def mbti_color(value: str) -> str:
    pole = decision_pole(value)
    if pole == "F":
        return F_COLOR
    if pole == "T":
        return T_COLOR
    return MODEL_COLOR


def build_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=220)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.0,
        1.03,
        "MBTI Outcomes Under F/T-Guided Tuning",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=21,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.0,
        0.965,
        "Cell color encodes the decision pole: F = Feeling, T = Thinking.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        color="#475569",
    )

    cell_text = df.values.tolist()
    col_labels = df.columns.tolist()
    cell_colors = [
        [MODEL_COLOR] + [mbti_color(value) for value in row[1:]]
        for row in cell_text
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.38, 0.19, 0.215, 0.215],
        bbox=[0.0, 0.18, 1.0, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(1.2)
        if row_idx == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(14)
            cell.set_height(0.13)
        else:
            cell.get_text().set_color(TEXT_COLOR)
            cell.get_text().set_weight("bold" if col_idx > 0 else "semibold")
            cell.set_height(0.15)
            if col_idx == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.08

    legend_handles = [
        mpatches.Patch(facecolor=F_COLOR, edgecolor=GRID_COLOR, label="F pole"),
        mpatches.Patch(facecolor=T_COLOR, edgecolor=GRID_COLOR, label="T pole"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.04),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=2.0,
    )

    ax.text(
        1.0,
        0.065,
        "Source: mbti_results/ft_dimension_mbti_summary.csv",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9.5,
        color="#64748B",
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    build_figure(df)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
