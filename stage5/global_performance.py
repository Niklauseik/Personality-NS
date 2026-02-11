# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_ORDER: list[str] = ["energy", "information", "decision", "execution", "ST-NF"]
MODEL_ORDER_BY_PAIR: dict[str, list[str]] = {
    "energy": ["E", "I"],
    "information": ["N", "S"],
    "decision": ["F", "T"],
    "execution": ["J", "P"],
    "ST-NF": ["ST", "NF"],
}
DATASET_ORDER: list[str] = ["imdb", "imdb_sklearn", "sst2", "fiqasa", "news", "mental"]


DOMAIN_BY_DATASET: dict[str, str] = {
    "imdb": "movie",
    "imdb_sklearn": "movie",
    "sst2": "movie",
    "fiqasa": "finance",
    "news": "finance",
    "mental": "mental",
}


def _clean_fieldname(name: str) -> str:
    return name.lstrip("\ufeff").strip()


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


def _bucket(delta: float | None, threshold: float) -> str:
    if delta is None or math.isnan(delta):
        return "unknown"
    if delta >= threshold:
        return "improve"
    if delta <= -threshold:
        return "decline"
    return "tie"


def _winner(adv: float | None, model_a: str, model_b: str, threshold: float) -> str:
    if adv is None or math.isnan(adv):
        return "unknown"
    if adv >= threshold:
        return model_a
    if adv <= -threshold:
        return model_b
    return "tie"


def _pair_sort_key(pair: str) -> tuple[int, str]:
    if pair in PAIR_ORDER:
        return (PAIR_ORDER.index(pair), pair)
    return (10**9, pair)


def _model_sort_key(pair: str, model: str) -> tuple[int, str]:
    order = MODEL_ORDER_BY_PAIR.get(pair)
    if order and model in order:
        return (order.index(model), model)
    return (10**9, model)


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    if dataset in DATASET_ORDER:
        return (DATASET_ORDER.index(dataset), dataset)
    return (10**9, dataset)


def _choose_model_order(pair: str, models: list[str]) -> tuple[str, str] | None:
    if len(models) != 2:
        return None
    preferred = MODEL_ORDER_BY_PAIR.get(pair)
    if preferred and set(models) == set(preferred):
        return preferred[0], preferred[1]
    ordered = sorted(models)
    return ordered[0], ordered[1]


def collect_sentiment_summary_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*_newlayout/*/summaries/sentiment.csv"))
    return sorted(paths)


def _read_sentiment_csv(csv_path: Path, root: Path) -> pd.DataFrame:
    rel = csv_path.relative_to(root) if csv_path.is_absolute() else csv_path
    model_root = rel.parts[0] if rel.parts else ""

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df.rename(columns={c: _clean_fieldname(c) for c in df.columns})
    for col in ["pair", "dataset", "run", "model"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["model_root"] = model_root
    df["source_file"] = rel.as_posix()

    for col in ["accuracy_strict", "f1_macro_strict", "n_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@dataclass(frozen=True)
class GlobalPerformanceOutputs:
    performance_long_csv: Path
    performance_pairwise_csv: Path
    performance_summary_csv: Path
    plots_dir: Path | None


def build_performance_tables(
    root: Path,
    run: str = "avg",
    threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = collect_sentiment_summary_paths(root)
    if not paths:
        raise FileNotFoundError("No sentiment.csv files found under *_newlayout/*/summaries/")

    all_rows = [_read_sentiment_csv(p, root) for p in paths]
    df = pd.concat(all_rows, ignore_index=True)
    required_cols = {"model_root", "pair", "dataset", "run", "model", "accuracy_strict", "f1_macro_strict"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in sentiment.csv: {sorted(missing)}")

    df = df.loc[df["run"].astype(str) == str(run)].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for run={run!r}.")

    key_cols = ["model_root", "pair", "dataset", "run"]
    metrics_cols = ["accuracy_strict", "f1_macro_strict"]
    keep_cols = key_cols + ["model", "source_file"]
    if "n_total" in df.columns:
        keep_cols.append("n_total")
    for col in metrics_cols:
        keep_cols.append(col)

    base_df = (
        df.loc[df["model"] == "BASE", key_cols + metrics_cols]
        .rename(
            columns={
                "accuracy_strict": "base_accuracy_strict",
                "f1_macro_strict": "base_f1_macro_strict",
            }
        )
        .copy()
    )

    model_df = df.loc[df["model"] != "BASE", keep_cols].copy()
    long_df = model_df.merge(base_df, on=key_cols, how="left")

    long_df["domain"] = long_df["dataset"].astype(str).map(DOMAIN_BY_DATASET).fillna("unknown")
    long_df["delta_accuracy_strict"] = long_df["accuracy_strict"] - long_df["base_accuracy_strict"]
    long_df["delta_f1_macro_strict"] = long_df["f1_macro_strict"] - long_df["base_f1_macro_strict"]
    long_df["delta_acc_bucket"] = long_df["delta_accuracy_strict"].map(lambda x: _bucket(x, threshold))
    long_df["delta_f1_bucket"] = long_df["delta_f1_macro_strict"].map(lambda x: _bucket(x, threshold))

    # Pairwise: per model_root/pair/dataset, compare two tuned models.
    pair_rows: list[dict[str, object]] = []
    pair_keys = ["model_root", "pair", "dataset", "domain", "run"]
    for (model_root, pair, dataset, domain, run_name), group in long_df.groupby(pair_keys, dropna=False):
        models = [str(m) for m in group["model"].tolist()]
        if len(models) != 2:
            continue
        chosen = _choose_model_order(str(pair), models)
        if chosen is None:
            continue
        model_a, model_b = chosen
        row_a = group.loc[group["model"] == model_a].iloc[0]
        row_b = group.loc[group["model"] == model_b].iloc[0]

        adv_acc = float(row_a["accuracy_strict"]) - float(row_b["accuracy_strict"])
        adv_f1 = float(row_a["f1_macro_strict"]) - float(row_b["f1_macro_strict"])
        pair_rows.append(
            {
                "model_root": model_root,
                "pair": pair,
                "dataset": dataset,
                "domain": domain,
                "run": run_name,
                "model_a": model_a,
                "model_b": model_b,
                "acc_a": float(row_a["accuracy_strict"]),
                "acc_b": float(row_b["accuracy_strict"]),
                "f1_a": float(row_a["f1_macro_strict"]),
                "f1_b": float(row_b["f1_macro_strict"]),
                "adv_acc_a_minus_b": adv_acc,
                "adv_f1_a_minus_b": adv_f1,
                "winner_acc": _winner(adv_acc, model_a, model_b, threshold),
                "winner_f1": _winner(adv_f1, model_a, model_b, threshold),
            }
        )

    pairwise_df = pd.DataFrame(pair_rows)

    # Summary: per model_root/pair/model/domain.
    def _agg(group: pd.DataFrame) -> pd.Series:
        out: dict[str, object] = {}
        deltas_acc = group["delta_accuracy_strict"].dropna().astype(float)
        deltas_f1 = group["delta_f1_macro_strict"].dropna().astype(float)

        out["n_datasets"] = int(group["dataset"].nunique())
        out["n_improve_acc"] = int((group["delta_acc_bucket"] == "improve").sum())
        out["n_decline_acc"] = int((group["delta_acc_bucket"] == "decline").sum())
        out["n_tie_acc"] = int((group["delta_acc_bucket"] == "tie").sum())
        out["improve_rate_acc"] = float(out["n_improve_acc"] / out["n_datasets"]) if out["n_datasets"] else math.nan
        out["mean_delta_acc"] = float(deltas_acc.mean()) if not deltas_acc.empty else math.nan
        out["median_delta_acc"] = float(deltas_acc.median()) if not deltas_acc.empty else math.nan

        out["n_improve_f1"] = int((group["delta_f1_bucket"] == "improve").sum())
        out["n_decline_f1"] = int((group["delta_f1_bucket"] == "decline").sum())
        out["n_tie_f1"] = int((group["delta_f1_bucket"] == "tie").sum())
        out["improve_rate_f1"] = float(out["n_improve_f1"] / out["n_datasets"]) if out["n_datasets"] else math.nan
        out["mean_delta_f1"] = float(deltas_f1.mean()) if not deltas_f1.empty else math.nan
        out["median_delta_f1"] = float(deltas_f1.median()) if not deltas_f1.empty else math.nan
        return pd.Series(out)

    summary_df = (
        long_df.groupby(["model_root", "pair", "model", "domain"], dropna=False)
        .apply(_agg)
        .reset_index()
    )
    for col in [
        "n_datasets",
        "n_improve_acc",
        "n_decline_acc",
        "n_tie_acc",
        "n_improve_f1",
        "n_decline_f1",
        "n_tie_f1",
    ]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").astype("Int64")

    long_df = long_df.sort_values(
        by=[
            "model_root",
            "pair",
            "model",
            "dataset",
        ],
        key=lambda s: s,
    )
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(
            by=["model_root", "pair", "dataset"],
            key=lambda s: s,
        )
    summary_df = summary_df.sort_values(by=["model_root", "pair", "model", "domain"], key=lambda s: s)

    return long_df, pairwise_df, summary_df


def _plot_heatmap(
    df_long: pd.DataFrame,
    out_path: Path,
    value_col: str,
    title: str,
) -> None:
    if df_long.empty:
        return
    pivot = df_long.pivot_table(index=["pair", "model"], columns="dataset", values=value_col, aggfunc="mean")
    row_index = sorted(pivot.index.tolist(), key=lambda x: (_pair_sort_key(str(x[0])), _model_sort_key(str(x[0]), str(x[1]))))
    col_index = sorted(pivot.columns.tolist(), key=lambda x: _dataset_sort_key(str(x)))
    pivot = pivot.reindex(index=row_index, columns=col_index)

    matrix = pivot.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.0
    max_abs = max(max_abs, 0.01)

    fig, ax = plt.subplots(figsize=(0.65 * len(col_index) + 4, 0.5 * len(row_index) + 3))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(col_index)))
    ax.set_xticklabels([str(c) for c in col_index], rotation=45, ha="right")
    ax.set_yticks(range(len(row_index)))
    ax.set_yticklabels([f"{pair}/{model}" for pair, model in row_index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_pushpull(
    df_pair_long: pd.DataFrame,
    out_path: Path,
    value_col: str,
    threshold: float,
    title: str,
) -> None:
    if df_pair_long.empty:
        return

    models = sorted(df_pair_long["model"].astype(str).unique().tolist())
    chosen = _choose_model_order(str(df_pair_long["pair"].iloc[0]), models)
    if chosen is None:
        return
    model_a, model_b = chosen

    pivot = df_pair_long.pivot_table(index="dataset", columns="model", values=value_col, aggfunc="mean")
    if model_a not in pivot.columns or model_b not in pivot.columns:
        return

    domain_colors = {"movie": "#1f77b4", "finance": "#ff7f0e", "mental": "#2ca02c", "unknown": "#7f7f7f"}
    datasets = sorted(pivot.index.tolist(), key=lambda x: _dataset_sort_key(str(x)))

    x = pivot[model_a].reindex(datasets).to_numpy(dtype=float)
    y = pivot[model_b].reindex(datasets).to_numpy(dtype=float)
    domains = df_pair_long.drop_duplicates(subset=["dataset"]).set_index("dataset")["domain"].reindex(datasets).tolist()

    max_abs = float(np.nanmax(np.abs(np.concatenate([x, y])))) if np.isfinite(x).any() or np.isfinite(y).any() else 0.0
    max_abs = max(max_abs, threshold * 2, 0.01)
    lim = max_abs * 1.15

    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.axvline(0.0, color="#888888", linewidth=1)
    for t in [threshold, -threshold]:
        ax.axhline(t, color="#cccccc", linewidth=1, linestyle="--")
        ax.axvline(t, color="#cccccc", linewidth=1, linestyle="--")

    for ds, xi, yi, dom in zip(datasets, x, y, domains):
        color = domain_colors.get(str(dom), domain_colors["unknown"])
        ax.scatter([xi], [yi], color=color, s=70, edgecolors="white", linewidth=0.6)
        ax.text(xi, yi, str(ds), fontsize=9, ha="left", va="bottom")

    handles = []
    labels = []
    for dom, color in domain_colors.items():
        if dom in set(domains):
            handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8))
            labels.append(dom)
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper left")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"Δ ({model_a} - BASE) [{value_col}]")
    ax.set_ylabel(f"Δ ({model_b} - BASE) [{value_col}]")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_domain_bar(
    df_pair_long: pd.DataFrame,
    out_path: Path,
    value_col: str,
    title: str,
) -> None:
    if df_pair_long.empty:
        return

    pair = str(df_pair_long["pair"].iloc[0])
    models = sorted(df_pair_long["model"].astype(str).unique().tolist())
    chosen = _choose_model_order(pair, models)
    if chosen is None:
        return
    model_a, model_b = chosen

    domain_order = ["movie", "finance", "mental"]
    domains_present = [d for d in domain_order if d in set(df_pair_long["domain"].tolist())]
    if not domains_present:
        return

    agg = (
        df_pair_long.groupby(["domain", "model"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    def get(domain: str, model: str) -> tuple[float, float]:
        row = agg.loc[(agg["domain"] == domain) & (agg["model"] == model)]
        if row.empty:
            return math.nan, 0.0
        mean = float(row["mean"].iloc[0])
        std = float(row["std"].iloc[0]) if not pd.isna(row["std"].iloc[0]) else 0.0
        return mean, std

    means_a, stds_a = zip(*(get(d, model_a) for d in domains_present), strict=True)
    means_b, stds_b = zip(*(get(d, model_b) for d in domains_present), strict=True)

    x = np.arange(len(domains_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - width / 2, means_a, width, label=model_a, yerr=stds_a, capsize=3)
    ax.bar(x + width / 2, means_b, width, label=model_b, yerr=stds_b, capsize=3)
    ax.axhline(0.0, color="#888888", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(domains_present)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_global_performance(
    root: Path,
    output_dir: Path,
    run: str = "avg",
    threshold: float = 0.005,
    plot: bool = False,
) -> GlobalPerformanceOutputs:
    _configure_matplotlib()
    long_df, pairwise_df, summary_df = build_performance_tables(root=root, run=run, threshold=threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "performance_long.csv"
    pairwise_path = output_dir / "performance_pairwise.csv"
    summary_path = output_dir / "performance_summary.csv"

    long_cols = [
        "model_root",
        "pair",
        "dataset",
        "domain",
        "run",
        "model",
        "base_accuracy_strict",
        "base_f1_macro_strict",
        "accuracy_strict",
        "f1_macro_strict",
        "delta_accuracy_strict",
        "delta_f1_macro_strict",
        "delta_acc_bucket",
        "delta_f1_bucket",
        "n_total",
        "source_file",
    ]
    long_cols = [col for col in long_cols if col in long_df.columns]
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig", columns=long_cols)

    pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plots_dir: Path | None = None
    if plot:
        plots_dir = output_dir / "plots"
        for model_root in sorted(long_df["model_root"].astype(str).unique().tolist()):
            model_rows = long_df.loc[long_df["model_root"] == model_root].copy()
            model_dir = plots_dir / model_root

            _plot_heatmap(
                model_rows,
                model_dir / "delta_accuracy_heatmap.png",
                value_col="delta_accuracy_strict",
                title=f"{model_root}: ΔAccuracy (model - BASE) [run={run}]",
            )
            _plot_heatmap(
                model_rows,
                model_dir / "delta_f1_heatmap.png",
                value_col="delta_f1_macro_strict",
                title=f"{model_root}: ΔMacro-F1 (model - BASE) [run={run}]",
            )

            for pair in sorted(model_rows["pair"].astype(str).unique().tolist(), key=_pair_sort_key):
                pair_rows = model_rows.loc[model_rows["pair"] == pair].copy()
                if pair_rows.empty:
                    continue
                _plot_pushpull(
                    pair_rows,
                    model_dir / f"pushpull_accuracy_{pair}.png",
                    value_col="delta_accuracy_strict",
                    threshold=threshold,
                    title=f"{model_root}/{pair}: Push-Pull (ΔAccuracy) [run={run}, t={threshold}]",
                )
                _plot_pushpull(
                    pair_rows,
                    model_dir / f"pushpull_f1_{pair}.png",
                    value_col="delta_f1_macro_strict",
                    threshold=threshold,
                    title=f"{model_root}/{pair}: Push-Pull (ΔMacro-F1) [run={run}, t={threshold}]",
                )
                _plot_domain_bar(
                    pair_rows,
                    model_dir / f"domain_bar_accuracy_{pair}.png",
                    value_col="delta_accuracy_strict",
                    title=f"{model_root}/{pair}: Mean ΔAccuracy by domain (std across datasets)",
                )
                _plot_domain_bar(
                    pair_rows,
                    model_dir / f"domain_bar_f1_{pair}.png",
                    value_col="delta_f1_macro_strict",
                    title=f"{model_root}/{pair}: Mean ΔMacro-F1 by domain (std across datasets)",
                )

    return GlobalPerformanceOutputs(
        performance_long_csv=long_path,
        performance_pairwise_csv=pairwise_path,
        performance_summary_csv=summary_path,
        plots_dir=plots_dir,
    )
