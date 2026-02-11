from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from scipy.stats import binomtest, chi2
except Exception:  # pragma: no cover - scipy not available
    binomtest = None
    chi2 = None


INVALID_TOKENS = {"invalid", "mixed"}


def _clean_alpha(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text.strip().lower() if "a" <= ch <= "z")


def _extract_pred_label(text: str, allowed: list[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return "invalid"
    text_l = text.lower()
    candidates = list(allowed) + ["neutral", "mixed"]
    earliest = None
    earliest_pos = 10**12
    for lbl in candidates:
        idx = text_l.find(lbl)
        if idx != -1 and idx < earliest_pos:
            earliest = lbl
            earliest_pos = idx
    if earliest is None:
        return "invalid"
    return _clean_alpha(earliest) or "invalid"


def _label_order_for_significance(dataset_key: str, allowed: list[str]) -> list[str]:
    allowed_clean = [_clean_alpha(x) for x in allowed]
    if dataset_key == "mental":
        return [x for x in allowed_clean if x]
    ordered: list[str] = []
    for label in ["positive", "negative", "neutral"]:
        if label in allowed_clean and label not in ordered:
            ordered.append(label)
    for label in allowed_clean:
        if label and label not in ordered:
            ordered.append(label)
    if "neutral" not in ordered:
        ordered.append("neutral")
    return ordered


def _normalize_prediction(text: str, allowed: list[str], dataset_key: str) -> str | None:
    label = _extract_pred_label(text, allowed)
    if dataset_key == "mental":
        return label if label in allowed else None
    if label in allowed:
        return label
    if label in INVALID_TOKENS or label == "neutral":
        return "neutral"
    return "neutral"


def _load_predictions(
    csv_path: Path,
    pred_col: str,
    allowed: list[str],
    dataset_key: str,
) -> list[str | None]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    preds = df[pred_col].astype(str).tolist()
    return [_normalize_prediction(p, allowed, dataset_key) for p in preds]


def _pick_prediction_path(csv_path: Path) -> Path | None:
    invalid_marker = csv_path.with_suffix(".invalid.csv")
    relabeled = csv_path.with_suffix(".relabeled.csv")
    processed = csv_path.with_suffix(".processed.csv")
    if invalid_marker.exists():
        if relabeled.exists():
            return relabeled
        if processed.exists():
            return processed
        return None
    return csv_path if csv_path.exists() else None


def _pair_labels(
    base_labels: list[str | None],
    tuned_labels: list[str | None],
) -> tuple[list[str], list[str], int, int]:
    total = min(len(base_labels), len(tuned_labels))
    kept_base: list[str] = []
    kept_tuned: list[str] = []
    dropped = 0
    for i in range(total):
        b = base_labels[i]
        t = tuned_labels[i]
        if b is None or t is None:
            dropped += 1
            continue
        kept_base.append(b)
        kept_tuned.append(t)
    return kept_base, kept_tuned, total, dropped


def _build_contingency(
    base_labels: list[str],
    tuned_labels: list[str],
    label_order: list[str],
) -> np.ndarray:
    label_to_idx = {lbl: idx for idx, lbl in enumerate(label_order)}
    counts = np.zeros((len(label_order), len(label_order)), dtype=int)
    for b, t in zip(base_labels, tuned_labels):
        if b not in label_to_idx or t not in label_to_idx:
            continue
        counts[label_to_idx[b], label_to_idx[t]] += 1
    return counts


def _cramers_v(stat: float | None, n_used: int, k: int) -> float | None:
    if stat is None or n_used <= 0 or k <= 1:
        return None
    denom = float(n_used) * float(k - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(float(stat) / denom))


def _mcnemar_effect_stat(counts: np.ndarray) -> float | None:
    if counts.shape != (2, 2):
        return None
    n01 = float(counts[0, 1])
    n10 = float(counts[1, 0])
    n = n01 + n10
    if n <= 0:
        return 0.0
    return float((n01 - n10) ** 2 / n)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _marginal_distributions(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(counts.sum())
    if total <= 0:
        k = counts.shape[0]
        return np.zeros(k, dtype=float), np.zeros(k, dtype=float)
    row = counts.sum(axis=1).astype(float) / total
    col = counts.sum(axis=0).astype(float) / total
    return row, col


def _chi2_sf(stat: float, df: int) -> float | None:
    if chi2 is None:
        return None
    return float(chi2.sf(stat, df))


def _mcnemar_test(counts: np.ndarray) -> tuple[float | None, int, float | None, str]:
    if counts.shape != (2, 2):
        return None, 1, None, "mcnemar"
    n01 = int(counts[0, 1])
    n10 = int(counts[1, 0])
    n = n01 + n10
    if n == 0:
        return 0.0, 1, 1.0, "mcnemar_exact"
    if binomtest is not None:
        p_value = float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
        return float(n), 1, p_value, "mcnemar_exact"
    stat = (abs(n01 - n10) - 1.0) ** 2 / float(n) if n > 0 else 0.0
    return float(stat), 1, _chi2_sf(stat, 1), "mcnemar_chi2"


def _stuart_maxwell_test(counts: np.ndarray) -> tuple[float | None, int, float | None, str]:
    k = counts.shape[0]
    if k <= 2:
        return None, max(k - 1, 1), None, "stuart_maxwell"
    row_sums = counts.sum(axis=1).astype(float)
    col_sums = counts.sum(axis=0).astype(float)
    d = row_sums - col_sums
    v = np.zeros((k - 1, k - 1), dtype=float)
    for i in range(k - 1):
        for j in range(k - 1):
            if i == j:
                v[i, j] = row_sums[i] + col_sums[i] - 2.0 * float(counts[i, i])
            else:
                v[i, j] = -float(counts[i, j] + counts[j, i])
    d_vec = d[: k - 1].reshape(-1, 1)
    try:
        v_inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:  # pragma: no cover - rare singular case
        v_inv = np.linalg.pinv(v)
    stat = float((d_vec.T @ v_inv @ d_vec).item())
    df = k - 1
    return stat, df, _chi2_sf(stat, df), "stuart_maxwell"


def evaluate_significance_newlayout(
    pair_root: Path,
    base_root: Path,
    datasets: dict[str, Any],
    model_codes: list[str],
    runs: list[str],
    output_path: Path,
    dataset_keys: list[str] | None = None,
) -> None:
    output_path = Path(output_path)
    pair_root = Path(pair_root)
    base_root = Path(base_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_keys = dataset_keys or list(datasets.keys())
    rows: list[dict] = []

    for dataset_key in dataset_keys:
        if dataset_key not in datasets:
            continue
        ds = datasets[dataset_key]
        allowed = [_clean_alpha(x) for x in ds.allowed_labels]
        label_order = _label_order_for_significance(dataset_key, allowed)
        if len(label_order) < 2:
            continue

        base_csv = _pick_prediction_path(
            base_root / "sentiment" / "run-001" / ds.dataset_dir / ds.filename
        )
        if base_csv is None:
            continue
        base_labels = _load_predictions(base_csv, ds.pred_col, allowed, dataset_key)

        for code in model_codes:
            model_dir = pair_root / code
            for run_name in runs:
                model_csv = _pick_prediction_path(
                    model_dir / "sentiment" / run_name / ds.dataset_dir / ds.filename
                )
                if model_csv is None:
                    continue
                tuned_labels = _load_predictions(model_csv, ds.pred_col, allowed, dataset_key)
                paired_base, paired_tuned, n_total, n_dropped = _pair_labels(base_labels, tuned_labels)
                if not paired_base:
                    continue
                counts = _build_contingency(paired_base, paired_tuned, label_order)
                row_dist, col_dist = _marginal_distributions(counts)
                tv = _tv_distance(row_dist, col_dist)
                js = _js_divergence(row_dist, col_dist)

                if counts.shape[0] == 2:
                    stat, df, p_value, test_name = _mcnemar_test(counts)
                    effect_stat = _mcnemar_effect_stat(counts)
                else:
                    stat, df, p_value, test_name = _stuart_maxwell_test(counts)
                    effect_stat = stat

                cramers_v = _cramers_v(effect_stat, len(paired_base), counts.shape[0])
                rows.append(
                    {
                        "pair": pair_root.name,
                        "dataset": dataset_key,
                        "run": run_name,
                        "model": code,
                        "test": test_name,
                        "statistic": stat,
                        "df": df,
                        "p_value": p_value,
                        "effect_cramers_v": cramers_v,
                        "effect_tv": tv,
                        "effect_js": js,
                        "n_total": n_total,
                        "n_used": len(paired_base),
                        "n_dropped": n_dropped,
                        "labels": "|".join(label_order),
                    }
                )

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        if output_path.exists():
            output_path.unlink()
