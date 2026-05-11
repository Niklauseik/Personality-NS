# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MIN_P = 1e-300
DEFAULT_P_THRESHOLD = 0.05
ANALYSIS_OUTPUT_DIR = "statistical_analysis"
RESERVED_DIRS = {"base", "summaries", "plots", "meta", ANALYSIS_OUTPUT_DIR}
NUMERIC_COLS = [
    "statistic",
    "df",
    "p_value",
    "effect_cramers_v",
    "effect_tv",
    "effect_js",
    "n_total",
    "n_used",
    "n_dropped",
]


def _discover_model_roots(patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        roots.extend(sorted(Path.cwd().glob(pattern)))
    return roots


def _is_model_root(path: Path) -> bool:
    return path.is_dir() and (path / "base").is_dir()


def _unique_existing_roots(roots: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _dimension_order(dims: list[str]) -> list[str]:
    preferred = ["energy", "information", "decision", "execution", "ST-NF", "NF-ST"]
    ordered: list[str] = []
    for name in preferred:
        if name in dims and name not in ordered:
            ordered.append(name)
    for name in sorted(dims):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang HK",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _load_significance_csv(dim_root: Path) -> pd.DataFrame | None:
    path = dim_root / "summaries" / "sentiment_significance.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clamp_p(p: float | None, min_p: float) -> float | None:
    if p is None or not np.isfinite(p):
        return None
    return float(max(p, min_p))


def _build_long_table(model_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in RESERVED_DIRS:
            continue
        df = _load_significance_csv(child)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["model_root"] = model_root.name
        df["dimension"] = child.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    long_df = _to_numeric(long_df, NUMERIC_COLS)
    return long_df


def _build_summary_table(long_df: pd.DataFrame, p_threshold: float, min_p: float) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    group_cols = ["model_root", "dimension", "dataset", "model"]
    agg_map = {col: "mean" for col in NUMERIC_COLS if col in long_df.columns}
    meta_cols = {"test": "first", "labels": "first"}
    agg_map.update({k: v for k, v in meta_cols.items() if k in long_df.columns})
    summary = long_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    summary["p_value_clamped"] = summary["p_value"].map(lambda x: _clamp_p(x, min_p))
    summary["neg_log10_p"] = summary["p_value_clamped"].map(
        lambda x: (-math.log10(x)) if x is not None and x > 0 else np.nan
    )
    summary["significant"] = summary["p_value"].map(
        lambda x: bool(x < p_threshold) if x is not None and np.isfinite(x) else False
    )
    summary["conclusion"] = summary["significant"].map(lambda x: "显著差异" if x else "不显著")
    return summary


def _build_dimension_summary(summary_df: pd.DataFrame, min_p: float) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    group_cols = ["model_root", "dimension", "dataset"]
    agg_map = {
        "p_value": "mean",
        "effect_cramers_v": "mean",
        "effect_tv": "mean",
        "effect_js": "mean",
        "n_total": "mean",
        "n_used": "mean",
        "n_dropped": "mean",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in summary_df.columns}
    dim_summary = summary_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    dim_summary["p_value_clamped"] = dim_summary["p_value"].map(lambda x: _clamp_p(x, min_p))
    dim_summary["neg_log10_p"] = dim_summary["p_value_clamped"].map(
        lambda x: (-math.log10(x)) if x is not None and x > 0 else np.nan
    )
    return dim_summary


def _plot_heatmap(
    dim_summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if dim_summary.empty:
        return
    _configure_matplotlib()
    dims = _dimension_order(sorted(dim_summary["dimension"].dropna().unique().tolist()))
    datasets = sorted(dim_summary["dataset"].dropna().unique().tolist())
    pivot = dim_summary.pivot(index="dimension", columns="dataset", values="neg_log10_p")
    pivot = pivot.reindex(index=dims, columns=datasets)
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    fig_w = max(6.0, 1.2 * len(datasets))
    fig_h = max(3.0, 0.6 * len(dims))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(dims)))
    ax.set_yticklabels(dims)
    ax.set_title(title, pad=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(p)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_violin_fallback(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    dims = _dimension_order(sorted(df["dimension"].dropna().unique().tolist()))
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    if not dims or not datasets:
        return

    data: list[list[float]] = []
    positions: list[float] = []
    entry_dims: list[str] = []
    for i, dataset in enumerate(datasets):
        base = i * (len(dims) + 1)
        for j, dim in enumerate(dims):
            vals = df.loc[(df["dataset"] == dataset) & (df["dimension"] == dim), metric].dropna().tolist()
            if not vals:
                continue
            data.append(vals)
            positions.append(base + j)
            entry_dims.append(dim)

    if not data:
        return

    _configure_matplotlib()
    fig_w = max(6.5, 1.3 * len(datasets))
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(dims)))
    dim_to_color = {dim: colors[i] for i, dim in enumerate(dims)}
    for body, dim in zip(parts["bodies"], entry_dims):
        body.set_facecolor(dim_to_color.get(dim, "#cccccc"))
        body.set_edgecolor("#444444")
        body.set_alpha(0.75)

    if "cmedians" in parts:
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.0)

    centers = [i * (len(dims) + 1) + (len(dims) - 1) / 2 for i in range(len(datasets))]
    ax.set_xticks(centers)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title, pad=8)

    handles = [
        plt.Line2D([0], [0], color=dim_to_color[dim], lw=6, label=dim) for dim in dims if dim in dim_to_color
    ]
    if handles:
        ax.legend(handles=handles, title="dimension", frameon=False, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_violin(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    try:
        import seaborn as sns  # type: ignore

        _configure_matplotlib()
        datasets = sorted(df["dataset"].dropna().unique().tolist())
        dims = _dimension_order(sorted(df["dimension"].dropna().unique().tolist()))
        fig_w = max(6.5, 1.3 * df["dataset"].nunique())
        fig_h = 4.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.violinplot(
            data=df,
            x="dataset",
            y=metric,
            hue="dimension",
            order=datasets if datasets else None,
            hue_order=dims if dims else None,
            cut=0,
            inner="quartile",
            scale="width",
            ax=ax,
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel("dataset")
        ax.set_ylabel(metric)
        ax.legend(title="dimension", frameon=False, loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception:
        _plot_violin_fallback(df, metric, output_path, title)


def _write_plot_readme_root(root_dir: Path, model_roots: list[Path], min_p: float, p_threshold: float) -> None:
    readme_path = root_dir / "STATISTICAL_ANALYSIS_README.md"
    model_list = "\n".join([f"- `{p.name}/{ANALYSIS_OUTPUT_DIR}/`" for p in model_roots]) if model_roots else "- (no outputs)"
    content = f"""# Statistical Analysis Outputs

This file explains the summary tables and plots generated by the statistical analysis pipeline.

## Scope
- Processes only new-layout model roots that contain `base/`
- Writes one output directory under each processed model root: `{ANALYSIS_OUTPUT_DIR}/`

## Generated Output Locations
{model_list}

## Input Data
- Each dimension/pair directory should contain `summaries/sentiment_significance.csv` from result processing.

## Summary Tables
- `significance_long.csv`
  - Row-level data: dimension/pair x dataset x run x model
- `significance_summary.csv`
  - Averages the same dimension/pair, dataset, and model across runs
  - Includes `p_value`, `effect_*`, and `conclusion`
- `significance_dimension_summary.csv`
  - Further averages by dimension/pair and dataset
  - This table is the direct input to the heatmap

## Plot 1: p-value Heatmap
- File: `pvalue_heatmap.png`
- X-axis: dataset
- Y-axis: personality dimension/pair
- Color: `-log10(p)`, with p-values clamped at {min_p}
- Reference: `p < {p_threshold}` corresponds to `-log10(p) > {-math.log10(p_threshold):.2f}`

## Plot 2: Effect-size Violin Plots
- One plot is generated for each effect metric:
  - `effect_cramers_v`
  - `effect_tv`
  - `effect_js`
- X-axis: dataset
- Color: personality dimension/pair
- The violin shape shows variation across runs and model variants
"""
    readme_path.write_text(content, encoding="utf-8")


def _write_outputs(model_root: Path, long_df: pd.DataFrame, min_p: float, p_threshold: float) -> None:
    out_root = model_root / ANALYSIS_OUTPUT_DIR
    summary_dir = out_root / "summaries"
    plots_root = out_root / "plots"
    heatmap_dir = plots_root / "heatmap"
    effect_dir = plots_root / "effect"

    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    if long_df.empty:
        return

    long_path = summary_dir / "significance_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    summary_df = _build_summary_table(long_df, p_threshold=p_threshold, min_p=min_p)
    summary_path = summary_dir / "significance_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    dim_summary = _build_dimension_summary(summary_df, min_p=min_p)
    dim_summary_path = summary_dir / "significance_dimension_summary.csv"
    dim_summary.to_csv(dim_summary_path, index=False, encoding="utf-8-sig")

    heatmap_path = heatmap_dir / "pvalue_heatmap.png"
    _plot_heatmap(
        dim_summary,
        output_path=heatmap_path,
        title=f"{model_root.name} p-value heatmap (-log10(p))",
    )

    for metric in ["effect_cramers_v", "effect_tv", "effect_js"]:
        if metric not in long_df.columns:
            continue
        metric_df = long_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue
        plot_path = effect_dir / f"{metric}_violin.png"
        _plot_violin(
            metric_df,
            metric=metric,
            output_path=plot_path,
            title=f"{model_root.name} {metric} (violin)",
        ) 


def run(args) -> None:
    requested = [Path(p) for p in args.model_root] if args.model_root else []
    discovered = _discover_model_roots(args.model_glob)
    targets = _unique_existing_roots(requested + discovered)
    if not targets:
        targets = [p for p in Path.cwd().iterdir() if _is_model_root(p)]

    if args.dry_run:
        print("[Statistical analysis] Dry-run. Would process:")
        for target in targets:
            if _is_model_root(target):
                print(f"  - {target}")
            else:
                print(f"  - {target} (skip: not a new-layout model root)")
        return

    processed: list[Path] = []
    for target in targets:
        if not _is_model_root(target):
            print(f"[Statistical analysis] Skipping non-newlayout root: {target}")
            continue
        long_df = _build_long_table(target)
        if long_df.empty:
            print(f"[Statistical analysis] No significance CSV found under: {target}")
            continue
        _write_outputs(target, long_df, min_p=args.min_p, p_threshold=args.p_threshold)
        processed.append(target)
        print(f"[Statistical analysis] Completed: {target}")

    _write_plot_readme_root(Path.cwd(), processed, min_p=args.min_p, p_threshold=args.p_threshold)
