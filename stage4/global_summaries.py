# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


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


def _binomial_p_greater_half(k: int, n: int) -> float:
    """Exact one-sided binomial test: H0 pass rate <= 0.5, H1 pass rate > 0.5."""
    if n <= 0:
        return math.nan
    if k < 0 or k > n:
        raise ValueError(f"Invalid binomial count k={k}, n={n}")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def _format_p_value(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.3g}"


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


def write_behavioural_shift_reliability_table(root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    rows = collect_significance_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_labels = {
        "llama-3b_newlayout": "Llama-3.2-3B",
        "qwen-3b_newlayout": "Qwen2.5-3B",
        "qwen-7b_newlayout": "Qwen2.5-7B",
    }
    model_order = ["llama-3b_newlayout", "qwen-3b_newlayout", "qwen-7b_newlayout"]
    dimension_labels = {
        "energy": "Energy",
        "information": "Info.",
        "decision": "Decision",
        "execution": "Exec.",
    }
    dimension_order = ["energy", "information", "decision", "execution"]

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"k": 0, "n": 0})
    for row in rows:
        model_root = row.get("model_root", "")
        pair = row.get("pair", "")
        if model_root not in model_labels or pair not in dimension_labels:
            continue
        p_value = _to_float(row.get("p_value"))
        if p_value is None:
            continue
        counts[(model_root, pair)]["n"] += 1
        if p_value < 0.05:
            counts[(model_root, pair)]["k"] += 1

    table_rows: list[dict[str, str | int]] = []
    for model_root in model_order:
        row: dict[str, str | int] = {
            "model": model_labels[model_root],
            "overall_k": 0,
            "overall_n": 0,
        }
        for pair in dimension_order:
            k = counts[(model_root, pair)]["k"]
            n = counts[(model_root, pair)]["n"]
            row[f"{pair}_k"] = k
            row[f"{pair}_n"] = n
            row[pair] = _format_p_value(_binomial_p_greater_half(k, n))
            row["overall_k"] = int(row["overall_k"]) + k
            row["overall_n"] = int(row["overall_n"]) + n
        row["overall"] = _format_p_value(_binomial_p_greater_half(int(row["overall_k"]), int(row["overall_n"])))
        table_rows.append(row)

    overall_row: dict[str, str | int] = {"model": "Overall", "overall_k": 0, "overall_n": 0}
    for pair in dimension_order:
        k = sum(counts[(model_root, pair)]["k"] for model_root in model_order)
        n = sum(counts[(model_root, pair)]["n"] for model_root in model_order)
        overall_row[f"{pair}_k"] = k
        overall_row[f"{pair}_n"] = n
        overall_row[pair] = _format_p_value(_binomial_p_greater_half(k, n))
        overall_row["overall_k"] = int(overall_row["overall_k"]) + k
        overall_row["overall_n"] = int(overall_row["overall_n"]) + n
    overall_row["overall"] = _format_p_value(
        _binomial_p_greater_half(int(overall_row["overall_k"]), int(overall_row["overall_n"]))
    )
    table_rows.append(overall_row)

    csv_path = output_dir / "behavioural_shift_reliability_binomial_table.csv"
    fieldnames = ["model"]
    for pair in dimension_order:
        fieldnames.extend([pair, f"{pair}_k", f"{pair}_n"])
    fieldnames.extend(["overall", "overall_k", "overall_n"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    md_path = output_dir / "behavioural_shift_reliability_binomial_table.md"
    headers = ["Model", "Energy", "Info.", "Decision", "Exec.", "Overall"]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Each cell reports the exact one-sided binomial p-value for the count of significant paired tests "
            "at p < 0.05, testing H0: pass rate <= 0.5 against H1: pass rate > 0.5.\n\n"
        )
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |\n")
        for row in table_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["model"]),
                        str(row["energy"]),
                        str(row["information"]),
                        str(row["decision"]),
                        str(row["execution"]),
                        str(row["overall"]),
                    ]
                )
                + " |\n"
            )

    tex_path = output_dir / "behavioural_shift_reliability_binomial_table.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\caption{Statistical reliability of behavioural shifts. Each cell reports the exact one-sided binomial p-value for the count of significant paired tests at $p<0.05$, testing $H_0$: pass rate $\\leq 0.5$ against $H_1$: pass rate $>0.5$.}\n")
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write("Model & Energy & Info. & Decision & Exec. & Overall \\\\\n")
        handle.write("\\midrule\n")
        for row in table_rows[:-1]:
            handle.write(
                f"{row['model']} & {row['energy']} & {row['information']} & {row['decision']} & "
                f"{row['execution']} & {row['overall']} \\\\\n"
            )
        handle.write("\\midrule\n")
        row = table_rows[-1]
        handle.write(
            f"{row['model']} & {row['energy']} & {row['information']} & {row['decision']} & "
            f"{row['execution']} & {row['overall']} \\\\\n"
        )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")

    return csv_path, md_path, tex_path


def _median(values: list[float]) -> float:
    return median(values) if values else math.nan


def _build_heatmap(rows: list[dict[str, str]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

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
    import matplotlib.pyplot as plt

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
