#!/usr/bin/env python3
"""Summarize sentiment significance CSVs into aggregate tables.

Outputs:
  - summaries/significance_long.csv: row-level concatenation
  - summaries/significance_summary.csv: aggregated metrics per pair/model/dataset/test
"""
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
            cleaned = { _clean_fieldname(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items() }
            cleaned["source_file"] = str(csv_path)
            rows.append(cleaned)
        return rows


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    root = Path(".")
    paths = list(root.glob("*_newlayout/*/summaries/sentiment_significance.csv"))
    if not paths:
        raise SystemExit("No sentiment_significance.csv files found under *_newlayout/*/summaries/")

    long_rows: list[dict[str, str]] = []
    for path in paths:
        long_rows.extend(_read_rows(path))

    summaries_dir = Path("summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)

    long_path = summaries_dir / "significance_long.csv"
    if long_rows:
        with long_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in long_rows for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(long_rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in long_rows:
        key = (
            row.get("pair", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("test", ""),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, str | float | int]] = []
    for (pair, model, dataset, test), rows in sorted(grouped.items()):
        p_values = [p for p in (_to_float(r.get("p_value")) for r in rows) if p is not None]
        cramers_v = [v for v in (_to_float(r.get("effect_cramers_v")) for r in rows) if v is not None]
        effect_tv = [v for v in (_to_float(r.get("effect_tv")) for r in rows) if v is not None]
        effect_js = [v for v in (_to_float(r.get("effect_js")) for r in rows) if v is not None]
        sig_count = sum(1 for p in p_values if p < 0.05)
        p_min = min(p_values) if p_values else math.nan
        p_max = max(p_values) if p_values else math.nan
        p_med = median(p_values) if p_values else math.nan
        summary_rows.append(
            {
                "pair": pair,
                "model": model,
                "dataset": dataset,
                "test": test,
                "n_rows": len(rows),
                "n_sig_p_lt_0.05": sig_count,
                "sig_rate": (sig_count / len(rows)) if rows else math.nan,
                "p_min": p_min,
                "p_median": p_med,
                "p_max": p_max,
                "effect_cramers_v_median": median(cramers_v) if cramers_v else math.nan,
                "effect_tv_median": median(effect_tv) if effect_tv else math.nan,
                "effect_js_median": median(effect_js) if effect_js else math.nan,
            }
        )

    summary_path = summaries_dir / "significance_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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

    print(f"Wrote {long_path} ({len(long_rows)} rows)")
    print(f"Wrote {summary_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
