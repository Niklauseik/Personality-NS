# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np

# === 0) 最新统计结果（与你上条消息一致） =========================
normalized_results_with_invalid = {
    "imdb_sentiment": {
        "base": {"negative": 5345, "positive": 4624, "neutral": 15, "mixed": 15, "invalid": 1},
        "s":    {"negative": 5582, "positive": 4387, "neutral": 9,  "mixed": 21, "invalid": 1},
        "n":    {"negative": 5148, "positive": 4751, "neutral": 43, "mixed": 57, "invalid": 1},
    },
    "mental_sentiment": {
        "base": {"depression": 26878, "normal": 4552, "mixed": 43, "invalid": 274},
        "s":    {"depression": 28001, "normal": 3572, "mixed": 62, "invalid": 112},
        "n":    {"depression": 26365, "normal": 5045, "mixed": 10, "invalid": 327},
    },
    "news_sentiment": {
        "base": {"bearish": 3371, "bullish": 5005, "neutral": 3552, "mixed": 0, "invalid": 3},
        "s":    {"bearish": 3368, "bullish": 4637, "neutral": 3921, "mixed": 0, "invalid": 5},
        "n":    {"bearish": 2804, "bullish": 5443, "neutral": 3681, "mixed": 0, "invalid": 3},
    },
    "fiqasa_sentiment": {
        "base": {"negative": 621, "neutral": 135, "positive": 417, "mixed": 0, "invalid": 0},
        "s":    {"negative": 711, "neutral": 92,  "positive": 370, "mixed": 0, "invalid": 0},
        "n":    {"negative": 519, "neutral": 275, "positive": 379, "mixed": 0, "invalid": 0},
    },
    "imdb_sklearn": {
        "base": {"negative": 5289, "positive": 4710, "neutral": 0, "mixed": 0, "invalid": 1},
        "s":    {"negative": 5665, "positive": 4334, "neutral": 0, "mixed": 0, "invalid": 1},
        "n":    {"negative": 5161, "positive": 4838, "neutral": 0, "mixed": 0, "invalid": 1},
    },
    "sst2": {
        "base": {"negative": 5594, "positive": 3920, "neutral": 484, "mixed": 1, "invalid": 1},
        "s":    {"negative": 6252, "positive": 3275, "neutral": 471, "mixed": 0, "invalid": 2},
        "n":    {"negative": 4927, "positive": 4145, "neutral": 917, "mixed": 5, "invalid": 6},
    },
}

# === 1) 允许标签（其余并入 invalid） =========================
allowed_labels = {
    "imdb_sentiment":   {"negative", "positive"},
    "mental_sentiment": {"depression", "normal"},
    "news_sentiment":   {"bearish", "bullish", "neutral"},
    "fiqasa_sentiment": {"negative", "positive", "neutral"},
    "imdb_sklearn":     {"negative", "positive"},
    "sst2":             {"negative", "positive"},  # 其余(如 neutral/mixed)并入 invalid
}

# === 2) 把非允许标签并入 invalid =================================
for dataset, model_data in normalized_results_with_invalid.items():
    allow = allowed_labels.get(dataset, set())
    for _, counts in model_data.items():
        extra = 0
        for label in list(counts.keys()):
            if label not in allow and label != "invalid":
                extra += counts.pop(label)
        counts["invalid"] = counts.get("invalid", 0) + extra

# === 3) 绘图函数：三饼图 & 分组柱状图 ============================
os.makedirs("plots", exist_ok=True)

def plot_pies(dataset: str, model_data: dict):
    order = ["base", "s", "n"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    # 总标题上移；子标题与饼图更贴近
    fig.suptitle(f"{dataset} Prediction Distribution", fontsize=14, y=0.98)

    for idx, model in enumerate(order):
        data = model_data.get(model, {})
        labels = list(data.keys())
        sizes  = list(data.values())
        axs[idx].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140
        )
        axs[idx].axis('equal')
        axs[idx].set_title(model.upper(), pad=2)

    fig.subplots_adjust(top=0.82, wspace=0.25)
    fig.savefig(f"plots/{dataset}_prediction_distribution_pie.png", dpi=150)
    plt.close(fig)

def plot_bars(dataset: str, model_data: dict):
    import numpy as np
    order = ["base", "s", "n"]

    # 统一标签顺序：按字母序，且把 "invalid" 放最后
    all_labels = set()
    for m in order:
        all_labels.update(model_data.get(m, {}).keys())
    labels = sorted([lbl for lbl in all_labels if lbl != "invalid"]) + (["invalid"] if "invalid" in all_labels else [])

    x = np.arange(len(labels))

    # 更“长”更紧凑：加大高度，减小宽度；略增柱宽以压缩组内空隙
    fig, ax = plt.subplots(figsize=(9, 6))   # 原 12x5 -> 9x6：更高、更不扁，也更紧凑
    width = 0.3                              # 原 0.25 -> 0.30：组内更紧凑

    # 画三组柱：base、s、n
    for i, m in enumerate(order):
        counts = [model_data.get(m, {}).get(lbl, 0) for lbl in labels]
        ax.bar(x + (i-1)*width, counts, width, label=m.upper())

    # 轴与布局：减少边距、让整体更“瘦长”
    ax.set_title(f"{dataset} Prediction Distribution", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Count")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.margins(x=0.01)                       # 减少左右留白
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)  # 进一步收紧两端

    # 收紧四周留白
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(f"plots/{dataset}_prediction_distribution_bar.png", dpi=150)
    plt.close(fig)

# === 4) 为每个数据集同时画饼图 + 柱状图 ==========================
for dataset, model_data in normalized_results_with_invalid.items():
    plot_pies(dataset, model_data)
    plot_bars(dataset, model_data)

print("✅ 全部图表已输出到 ./plots 目录")
