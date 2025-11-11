# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ========= 需要读取的统计结果文件 =========
SUMMARY_TXT = "label_distribution_summary.txt"  # ← 改成你的实际路径

# ========= 每个数据集允许的标签（其余并入 invalid）=========
allowed_labels = {
    "imdb_sentiment":   {"negative", "positive"},
    "mental_sentiment": {"depression", "normal"},
    "news_sentiment":   {"bearish", "bullish", "neutral"},
    "fiqasa_sentiment": {"negative", "positive", "neutral"},
    "imdb_sklearn":     {"negative", "positive"},
    "sst2":             {"negative", "positive"},  # 其余(如 neutral/mixed)并入 invalid
}

# ========= 解析 txt 为 {dataset: {model: {label: count}}} =========
# txt 结构形如：
# ======== imdb_sentiment ========
#           真实数量  基座模型   S模型   N模型
# negative  5000     5331     5641     5134
# positive  5000     4636     4350     4694
# neutral      0       15        2       56
# mixed        0       17        7      100
# invalid      0        1        0       16
#
# （空行后进入下一个数据集）

section_pat = re.compile(r"^=+\s*(.+?)\s*=+\s*$")

def parse_summary_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary file not found: {path}")

    results = {}  # {dataset: {"base":{}, "s":{}, "n":{}}}

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        m = section_pat.match(line)
        if not m:
            i += 1
            continue

        dataset = m.group(1).strip()
        # 下一行应为表头
        i += 1
        if i >= n:
            break
        header = lines[i].strip()
        # 兼容中英列名；我们只关心三列：基座模型、S模型、N模型
        # 允许“真实数量/true”、“基座模型/base”、“S模型/s”、“N模型/n”
        cols = header.split()
        # 构造列名到索引的映射
        col_map = {name: idx for idx, name in enumerate(cols)}

        # 找需要的列索引（必须存在）
        def pick(*names):
            for nm in names:
                if nm in col_map:
                    return col_map[nm]
            return None

        idx_true = pick("真实数量", "true")
        idx_base = pick("基座模型", "base")
        idx_s    = pick("S模型", "s", "S")
        idx_n    = pick("N模型", "n", "N")

        if None in (idx_base, idx_s, idx_n):
            # 表头不规范时提示
            raise ValueError(f"Unrecognized header columns for dataset '{dataset}': {header}")

        # 初始化数据结构
        results.setdefault(dataset, {"base": {}, "s": {}, "n": {}})

        # 读取数据行直到遇到空行或下一节
        i += 1
        while i < n and lines[i].strip():
            row = lines[i].strip()
            # 按空白切分：第一列是 label，后面是数字列
            parts = row.split()
            if len(parts) < 2:
                i += 1
                continue

            label = parts[0]
            # 确保数字列长度足够
            # 由于 header 已经 split，parts 的列与 header 对齐
            def safe_int(j):
                try:
                    return int(parts[j])
                except Exception:
                    return 0

            # 读取三模型的计数
            base_val = safe_int(1 + idx_base)
            s_val    = safe_int(1 + idx_s)
            n_val    = safe_int(1 + idx_n)

            results[dataset]["base"][label] = base_val
            results[dataset]["s"][label]    = s_val
            results[dataset]["n"][label]    = n_val

            i += 1

        # 跳过本节后的空行
        while i < n and not lines[i].strip():
            i += 1

    return results

normalized_results_with_invalid = parse_summary_txt(SUMMARY_TXT)

# ========= 把非允许标签并入 invalid =========
for dataset, model_data in normalized_results_with_invalid.items():
    allow = allowed_labels.get(dataset, set())
    for _, counts in model_data.items():  # counts: {label:count}
        extra = 0
        for label in list(counts.keys()):
            if label not in allow and label != "invalid":
                extra += counts.pop(label)
        counts["invalid"] = counts.get("invalid", 0) + extra

# ========= 绘图（与你原始风格一致，略做紧凑处理） =========
os.makedirs("plots", exist_ok=True)

def plot_pies(dataset: str, model_data: dict):
    order = ["base", "s", "n"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"{dataset} Prediction Distribution", fontsize=14, y=0.98)

    for idx, model in enumerate(order):
        data = model_data.get(model, {})
        if not data:
            axs[idx].axis('off')
            axs[idx].set_title(model.upper(), pad=2)
            continue
        labels = list(data.keys())
        sizes  = list(data.values())
        axs[idx].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        axs[idx].axis('equal')
        axs[idx].set_title(model.upper(), pad=2)

    fig.subplots_adjust(top=0.82, wspace=0.25)
    fig.savefig(f"plots/{dataset}_prediction_distribution_pie.png", dpi=150)
    plt.close(fig)

def plot_bars(dataset: str, model_data: dict):
    order = ["base", "s", "n"]
    # 统一标签顺序：按字母序，且把 "invalid" 放最后
    all_labels = set()
    for m in order:
        all_labels.update(model_data.get(m, {}).keys())
    labels = sorted([lbl for lbl in all_labels if lbl != "invalid"]) + (["invalid"] if "invalid" in all_labels else [])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.3

    for i, m in enumerate(order):
        counts = [model_data.get(m, {}).get(lbl, 0) for lbl in labels]
        ax.bar(x + (i-1)*width, counts, width, label=m.upper())

    ax.set_title(f"{dataset} Prediction Distribution", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Count")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.margins(x=0.01)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(f"plots/{dataset}_prediction_distribution_bar.png", dpi=150)
    plt.close(fig)

# ========= 生成全部图表 =========
for dataset, model_data in normalized_results_with_invalid.items():
    plot_pies(dataset, model_data)
    plot_bars(dataset, model_data)

print("✅ 全部图表已输出到 ./plots 目录")
