# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .sentiment_significance import evaluate_significance_newlayout


@dataclass(frozen=True)
class SentimentDataset:
    dataset_dir: str
    filename: str
    label_col: str
    pred_col: str
    allowed_labels: list[str]
    label_map: dict[str, str] | None = None


SENTIMENT_DATASETS: dict[str, SentimentDataset] = {
    "imdb": SentimentDataset(
        dataset_dir="imdb",
        filename="imdb_sentiment_results.csv",
        label_col="label",
        pred_col="prediction",
        allowed_labels=["positive", "negative"],
        label_map={"0": "positive", "1": "negative"},
    ),
    "mental": SentimentDataset(
        dataset_dir="mental",
        filename="mental_sentiment_results.csv",
        label_col="label",
        pred_col="prediction",
        allowed_labels=["normal", "depression"],
        label_map=None,
    ),
    "news": SentimentDataset(
        dataset_dir="news",
        filename="news_sentiment_results.csv",
        label_col="label",
        pred_col="prediction",
        allowed_labels=["bearish", "bullish", "neutral"],
        label_map={"0": "bearish", "1": "bullish", "2": "neutral"},
    ),
    "fiqasa": SentimentDataset(
        dataset_dir="fiqasa",
        filename="fiqasa_sentiment_results.csv",
        label_col="answer",
        pred_col="prediction",
        allowed_labels=["negative", "positive", "neutral"],
        label_map=None,
    ),
    "imdb_sklearn": SentimentDataset(
        dataset_dir="imdb_sklearn",
        filename="imdb_sklearn_sentiment_results.csv",
        label_col="label",
        pred_col="prediction",
        allowed_labels=["negative", "positive"],
        label_map={"0": "negative", "1": "positive"},
    ),
    "sst2": SentimentDataset(
        dataset_dir="sst2",
        filename="sst2_sentiment_results.csv",
        label_col="label",
        pred_col="prediction",
        allowed_labels=["negative", "positive"],
        label_map={"0": "negative", "1": "positive"},
    ),
}

BENCHMARK_FILES: dict[str, str] = {
    "ARC (easy)": "arc_easy_test800_results.csv",
    "BoolQ": "boolq_train800_results.csv",
    "GSM8K": "gsm8k_test800_results.csv",
}

RESERVED_DIRS = {"summaries", "plots", "meta"}


def _order_labels(allowed: list[str]) -> list[str]:
    ordered: list[str] = []
    for label in ["positive", "negative", "neutral"]:
        if label in allowed and label not in ordered:
            ordered.append(label)
    for label in allowed:
        if label not in ordered:
            ordered.append(label)
    if "neutral" not in ordered:
        ordered.append("neutral")
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


def _clean_alpha(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z]", "", text.strip().lower())


def _map_true_labels(ds: SentimentDataset, series: pd.Series) -> pd.Series:
    if ds.label_map:
        mapped = series.astype(str).map(ds.label_map).fillna("")
        return mapped.astype(str).map(_clean_alpha)
    return series.astype(str).map(_clean_alpha)


def _extract_pred_label(text: str, allowed: list[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    text_l = text.lower()
    candidates = list(allowed) + ["neutral", "mixed"]
    earliest = None
    earliest_pos = 10**12
    for lbl in candidates:
        match = re.search(rf"\b{re.escape(lbl)}\b", text_l)
        if match and match.start() < earliest_pos:
            earliest = lbl
            earliest_pos = match.start()
    if earliest is None:
        return "invalid"
    return _clean_alpha(earliest) or "invalid"


def _compute_metrics(y_true: list[str], y_pred: list[str], class_labels: list[str]) -> dict[str, float | int]:
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "support": int(len(y_true)),
    }


def _mean_metrics(metrics_list: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not metrics_list:
        return {}
    keys = list(metrics_list[0].keys())
    out: dict[str, float | int] = {}
    for key in keys:
        if key == "support":
            out[key] = int(round(float(np.mean([float(m.get(key, 0)) for m in metrics_list]))))
        else:
            out[key] = float(np.mean([float(m.get(key, 0.0)) for m in metrics_list]))
    return out


def _iter_pair_model_codes(pair_root: Path) -> list[str]:
    codes: list[str] = []
    for child in sorted(pair_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in RESERVED_DIRS:
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]+", child.name) is None:
            continue
        codes.append(child.name)
    return codes


def _collect_run_names(model_dir: Path) -> list[str]:
    runs_root = model_dir / "sentiment"
    if not runs_root.exists():
        return []
    runs = sorted([p.name for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run-")])
    return runs


def _load_existing_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:  # pragma: no cover
        return None


def _summary_has_sentinel(
    df: pd.DataFrame | None,
    dataset_key: str,
    run_name: str,
    models: list[str],
) -> bool:
    if df is None or df.empty:
        return False
    required_cols = {"dataset", "run", "model"}
    if not required_cols.issubset(set(df.columns)):
        return False
    mask = (df["dataset"] == dataset_key) & (df["run"] == run_name)
    present = set(df.loc[mask, "model"].astype(str).tolist())
    return set(models).issubset(present)


def _normalize_existing_summary(summary_path: Path) -> pd.DataFrame | None:
    existing = _load_existing_summary(summary_path)
    if existing is None or existing.empty:
        return existing
    # If a legacy long-table summary exists, move it aside.
    if "record_type" in existing.columns:
        legacy_path = summary_path.with_name(summary_path.stem + "_long_backup.csv")
        if not legacy_path.exists():
            summary_path.rename(legacy_path)
        return None
    return existing


def _append_summary_rows(summary_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    existing = _normalize_existing_summary(summary_path)
    new_df = pd.DataFrame(rows)
    merged = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df

    key_cols = ["pair", "dataset", "run", "model"]
    for col in key_cols:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged.drop_duplicates(subset=key_cols, keep="last")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")


def _plot_distribution_bars(
    dataset_key: str,
    label_order: list[str],
    model_order: list[str],
    counts_by_model: dict[str, dict[str, float]],
    output_path: Path,
    title_suffix: str,
) -> None:
    _configure_matplotlib()
    x = np.arange(len(label_order))
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / max(len(model_order), 1)
    for i, model in enumerate(model_order):
        counts = [counts_by_model.get(model, {}).get(lbl, 0) for lbl in label_order]
        ax.bar(x + (i - (len(model_order) - 1) / 2) * width, counts, width, label=model)
    ax.set_title(f"{dataset_key} Prediction Distribution ({title_suffix})", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=15)
    ax.set_ylabel("Count")
    ax.legend(ncol=min(len(model_order), 3), frameon=False, loc="upper right")
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_distribution_pies(
    dataset_key: str,
    label_order: list[str],
    model_order: list[str],
    counts_by_model: dict[str, dict[str, float]],
    output_path: Path,
    title_suffix: str,
) -> None:
    _configure_matplotlib()
    fig, axs = plt.subplots(1, len(model_order), figsize=(4 * len(model_order), 5))
    fig.suptitle(f"{dataset_key} Prediction Distribution ({title_suffix})", fontsize=14, y=0.98)
    for idx, model in enumerate(model_order):
        ax = axs[idx] if len(model_order) > 1 else axs
        counts = [counts_by_model.get(model, {}).get(lbl, 0.0) for lbl in label_order]
        if sum(counts) <= 0:
            ax.axis("off")
            ax.set_title(model, pad=2)
            continue
        ax.pie(counts, labels=label_order, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        ax.set_title(model, pad=2)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def _row_from_eval_wide(
    pair_name: str,
    dataset_key: str,
    run_name: str,
    model: str,
    dist: dict[str, float],
    metrics: dict[str, dict[str, float | int]],
    n_total: int,
    label_order: list[str],
) -> dict:
    total = float(n_total) if n_total else float(sum(dist.values()))
    denom = float(total) if total else 1.0
    row: dict = {
        "pair": pair_name,
        "dataset": dataset_key,
        "run": run_name,
        "model": model,
        "labels": "|".join(label_order),
        "n_total": total,
    }
    for label in label_order:
        count = float(dist.get(label, 0.0))
        row[f"count_{label}"] = count
        row[f"ratio_{label}"] = float(count) / denom

    # flatten metrics into *_strict / *_neutral
    for variant, metric_map in metrics.items():
        for metric, value in metric_map.items():
            suffix = f"_{variant}"
            key = f"{metric}{suffix}"
            row[key] = int(value) if metric == "support" else float(value)
    return row


def _evaluate_sentiment_file(csv_path: Path, ds: SentimentDataset) -> tuple[dict[str, float], dict[str, dict], int]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    allowed = [_clean_alpha(x) for x in ds.allowed_labels]

    # Distribution over all rows (not just evaluable subset): invalid/mixed -> neutral.
    pred_extracted_all = df[ds.pred_col].astype(str).map(lambda x: _extract_pred_label(x, allowed)).tolist()
    label_order = _order_labels(allowed)
    dist = Counter({lbl: 0 for lbl in label_order})
    for p in pred_extracted_all:
        if p in allowed:
            dist[p] += 1
        else:
            dist["neutral"] += 1
    total_dist = int(sum(dist.values()))

    # Metrics on evaluable subset only (true label in allowed)
    y_true_all = _map_true_labels(ds, df[ds.label_col])
    kept = df.loc[y_true_all.isin(allowed)].copy()
    if kept.empty:
        return {k: int(v) for k, v in dist.items()}, {"strict": {}, "neutral": {}}, total_dist

    pred_extracted = kept[ds.pred_col].astype(str).map(lambda x: _extract_pred_label(x, allowed)).tolist()
    y_true = _map_true_labels(ds, kept[ds.label_col]).tolist()

    y_pred_strict = [p if p in allowed else "invalid" for p in pred_extracted]
    y_pred_neutral = ["neutral" if p in {"invalid", "mixed"} else p for p in y_pred_strict]

    metrics = {
        "strict": _compute_metrics(y_true, y_pred_strict, class_labels=allowed),
        "neutral": _compute_metrics(y_true, y_pred_neutral, class_labels=allowed),
    }
    dist_dict = {k: float(v) for k, v in dist.items()}
    return dist_dict, metrics, total_dist


def _evaluate_benchmark_pair(base_dir: Path, model_dirs: dict[str, Path], model_codes: list[str]) -> list[dict]:
    def extract_upper_letter(text):
        match = re.search(r"\b([A-D])\b", str(text).upper())
        return match.group(1) if match else None

    def extract_bool(text):
        if isinstance(text, str):
            t = text.lower()
            if "true" in t:
                return True
            if "false" in t:
                return False
        elif isinstance(text, bool):
            return text
        return None

    def extract_numbers(text):
        text = str(text).replace(",", "").replace("$", "")
        return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

    def gsm8k_accuracy_from_numbers(df):
        correct, total = 0, 0
        for _, row in df.iterrows():
            label_nums = extract_numbers(row.get("label", ""))
            pred_nums = extract_numbers(row.get("prediction", ""))
            if not label_nums or not pred_nums:
                continue
            if any(label in pred_nums for label in label_nums):
                correct += 1
            total += 1
        acc = correct / total if total else 0.0
        return float(acc), correct, total

    def compute_prf(y_true, y_pred):
        y_true_seq = list(y_true)
        y_pred_seq = list(y_pred)
        accuracy = float(accuracy_score(y_true_seq, y_pred_seq)) if y_true_seq else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_seq, y_pred_seq, average="macro", zero_division=0
        )
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    rows: list[dict] = []
    targets: list[tuple[str, Path]] = [("BASE", base_dir)] + [(c, model_dirs[c]) for c in model_codes]
    for model_name, model_path in targets:
        bench_dir = model_path / "benchmark"
        if not bench_dir.exists():
            continue
        for dataset_name, filename in BENCHMARK_FILES.items():
            file_path = bench_dir / filename
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path)
            if dataset_name == "ARC (easy)":
                df["label_clean"] = df["label"].apply(extract_upper_letter)
                df["prediction_clean"] = df["prediction"].apply(extract_upper_letter)
                df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
                metrics = compute_prf(df_valid["label_clean"], df_valid["prediction_clean"])
                extra = {}
            elif dataset_name == "BoolQ":
                df["label_clean"] = df["label"].apply(extract_bool)
                df["prediction_clean"] = df["prediction"].apply(extract_bool)
                df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
                metrics = compute_prf(df_valid["label_clean"], df_valid["prediction_clean"])
                extra = {}
            elif dataset_name == "GSM8K":
                acc, correct, total = gsm8k_accuracy_from_numbers(df)
                metrics = {"accuracy": acc, "precision": "", "recall": "", "f1": ""}
                extra = {"correct": int(correct), "total": int(total)}
            else:
                metrics = {"accuracy": "", "precision": "", "recall": "", "f1": ""}
                extra = {}
            rows.append({"model": model_name, "dataset": dataset_name, **metrics, **extra})
    return rows


def process_model_root(model_root: Path) -> None:
    model_root = Path(model_root)
    base_root = model_root / "base"
    if not base_root.exists():
        raise FileNotFoundError(f"Missing base folder: {base_root}")

    dimensions = [p for p in sorted(model_root.iterdir()) if p.is_dir() and p.name != "base"]
    if not dimensions:
        print(f"[Result processing] No dimensions/pairs found under: {model_root}")
        return

    for pair_root in dimensions:
        _process_pair_root(pair_root, base_root)


def _process_pair_root(pair_root: Path, base_root: Path) -> None:
    pair_name = pair_root.name
    model_codes = _iter_pair_model_codes(pair_root)
    if len(model_codes) < 2:
        print(f"[Result processing] Skipping {pair_root}: expected 2 model folders, found {len(model_codes)}.")
        return
    if len(model_codes) > 2:
        print(f"[Result processing] WARNING: {pair_root} has {len(model_codes)} model folders; using first 2: {model_codes[:2]}")
        model_codes = model_codes[:2]

    model_dirs = {code: (pair_root / code) for code in model_codes}

    runs_a = set(_collect_run_names(model_dirs[model_codes[0]]))
    runs_b = set(_collect_run_names(model_dirs[model_codes[1]]))
    runs = sorted(runs_a.intersection(runs_b))
    if not runs:
        print(f"[Result processing] Skipping {pair_root}: no shared sentiment runs found under both models.")
        return

    summary_dir = pair_root / "summaries"
    plots_root = pair_root / "plots"
    bar_dir = plots_root / "bar"
    pie_dir = plots_root / "pie"
    sentiment_summary_path = summary_dir / "sentiment.csv"
    benchmark_summary_path = summary_dir / "benchmark.csv"

    existing_summary = _normalize_existing_summary(sentiment_summary_path)
    all_models_for_plots = ["BASE"] + model_codes

    # benchmark once per pair
    if not benchmark_summary_path.exists():
        bench_rows = _evaluate_benchmark_pair(base_root, model_dirs, model_codes)
        if bench_rows:
            benchmark_summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(bench_rows).to_csv(benchmark_summary_path, index=False, encoding="utf-8-sig")

    # discover which datasets exist
    datasets_present: set[str] = set()
    for code in model_codes:
        for run_name in runs:
            run_root = model_dirs[code] / "sentiment" / run_name
            if not run_root.exists():
                continue
            datasets_present.update([p.name for p in run_root.iterdir() if p.is_dir()])

    datasets = [k for k in SENTIMENT_DATASETS.keys() if k in datasets_present]
    if not datasets:
        print(f"[Result processing] No known sentiment datasets found under: {pair_root}")
        return

    rows: list[dict] = []

    # base cache (fixed run-001)
    base_cache: dict[str, tuple[dict[str, float], dict[str, dict], int]] = {}
    for dataset_key in datasets:
        ds = SENTIMENT_DATASETS[dataset_key]
        base_csv = base_root / "sentiment" / "run-001" / ds.dataset_dir / ds.filename
        if not base_csv.exists():
            print(f"[Result processing] WARNING: missing base sentiment file: {base_csv}")
            continue
        base_cache[dataset_key] = _evaluate_sentiment_file(base_csv, ds)

    for dataset_key in datasets:
        if dataset_key not in base_cache:
            continue
        ds = SENTIMENT_DATASETS[dataset_key]
        label_order = _order_labels([_clean_alpha(x) for x in ds.allowed_labels])

        per_run_metrics: dict[str, dict[str, list[dict[str, float | int]]]] = {
            code: {"strict": [], "neutral": []} for code in model_codes
        }
        per_run_dist: dict[str, dict[str, dict[str, float]]] = {code: {} for code in model_codes}

        for run_name in runs:
            bar_path = bar_dir / f"{dataset_key}__{run_name}.png"
            pie_path = pie_dir / f"{dataset_key}__{run_name}.png"
            if (
                bar_path.exists()
                and pie_path.exists()
                and _summary_has_sentinel(existing_summary, dataset_key, run_name, all_models_for_plots)
            ):
                continue

            counts_by_model: dict[str, dict[str, float]] = {}
            base_dist, base_metrics, base_total = base_cache[dataset_key]
            counts_by_model["BASE"] = base_dist
            rows.append(
                _row_from_eval_wide(pair_name, dataset_key, run_name, "BASE", base_dist, base_metrics, base_total, label_order)
            )

            missing = False
            for code in model_codes:
                model_csv = model_dirs[code] / "sentiment" / run_name / ds.dataset_dir / ds.filename
                if not model_csv.exists():
                    print(f"[Result processing] WARNING: missing sentiment file: {model_csv}")
                    missing = True
                    continue
                dist, metrics, total = _evaluate_sentiment_file(model_csv, ds)
                counts_by_model[code] = dist
                rows.append(_row_from_eval_wide(pair_name, dataset_key, run_name, code, dist, metrics, total, label_order))
                per_run_dist[code][run_name] = dist
                per_run_metrics[code]["strict"].append(metrics["strict"])
                per_run_metrics[code]["neutral"].append(metrics["neutral"])

            if not missing and set(counts_by_model.keys()) == set(all_models_for_plots):
                _plot_distribution_bars(
                    dataset_key=dataset_key,
                    label_order=label_order,
                    model_order=all_models_for_plots,
                    counts_by_model=counts_by_model,
                    output_path=bar_path,
                    title_suffix=run_name,
                )
                _plot_distribution_pies(
                    dataset_key=dataset_key,
                    label_order=label_order,
                    model_order=all_models_for_plots,
                    counts_by_model=counts_by_model,
                    output_path=pie_path,
                    title_suffix=run_name,
                )

        # avg plot + avg summary
        avg_bar_path = bar_dir / f"{dataset_key}__avg.png"
        avg_pie_path = pie_dir / f"{dataset_key}__avg.png"
        if not (
            avg_bar_path.exists()
            and avg_pie_path.exists()
            and _summary_has_sentinel(existing_summary, dataset_key, "avg", all_models_for_plots)
        ):
            counts_by_model: dict[str, dict[str, float]] = {}
            base_dist, base_metrics, base_total = base_cache[dataset_key]
            counts_by_model["BASE"] = base_dist
            rows.append(_row_from_eval_wide(pair_name, dataset_key, "avg", "BASE", base_dist, base_metrics, base_total, label_order))

            missing = False
            for code in model_codes:
                # If per-run data wasn't collected (e.g., per-run step skipped), compute from files.
                if not per_run_dist[code]:
                    summed = Counter()
                    strict_list: list[dict[str, float | int]] = []
                    neutral_list: list[dict[str, float | int]] = []
                    run_count = 0
                    for run_name in runs:
                        model_csv = model_dirs[code] / "sentiment" / run_name / ds.dataset_dir / ds.filename
                        if not model_csv.exists():
                            missing = True
                            continue
                        dist, metrics, _ = _evaluate_sentiment_file(model_csv, ds)
                        summed.update(dist)
                        strict_list.append(metrics["strict"])
                        neutral_list.append(metrics["neutral"])
                        run_count += 1
                    per_run_dist[code] = {r: {} for r in runs} if missing else per_run_dist[code]
                    per_run_metrics[code]["strict"] = strict_list
                    per_run_metrics[code]["neutral"] = neutral_list
                else:
                    summed = Counter()
                    for dist in per_run_dist[code].values():
                        summed.update(dist)
                    run_count = len(per_run_dist[code])

                if run_count <= 0 or run_count != len(runs):
                    missing = True
                    continue
                dist_avg = {lbl: float(summed.get(lbl, 0.0)) / float(run_count) for lbl in label_order}
                total = float(sum(dist_avg.values()))
                counts_by_model[code] = dist_avg
                metrics_avg = {
                    "strict": _mean_metrics(per_run_metrics[code]["strict"]),
                    "neutral": _mean_metrics(per_run_metrics[code]["neutral"]),
                }
                rows.append(_row_from_eval_wide(pair_name, dataset_key, "avg", code, dist_avg, metrics_avg, total, label_order))

            if not missing and set(counts_by_model.keys()) == set(all_models_for_plots):
                _plot_distribution_bars(
                    dataset_key=dataset_key,
                    label_order=label_order,
                    model_order=all_models_for_plots,
                    counts_by_model=counts_by_model,
                    output_path=avg_bar_path,
                    title_suffix="avg",
                )
                _plot_distribution_pies(
                    dataset_key=dataset_key,
                    label_order=label_order,
                    model_order=all_models_for_plots,
                    counts_by_model=counts_by_model,
                    output_path=avg_pie_path,
                    title_suffix="avg",
                )

    _append_summary_rows(sentiment_summary_path, rows)
    evaluate_significance_newlayout(
        pair_root=pair_root,
        base_root=base_root,
        datasets=SENTIMENT_DATASETS,
        model_codes=model_codes,
        runs=runs,
        output_path=summary_dir / "sentiment_significance.csv",
        dataset_keys=datasets,
    )
