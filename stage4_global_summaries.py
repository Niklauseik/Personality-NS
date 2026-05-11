# -*- coding: utf-8 -*-
"""Stage-4 helper: build global summaries/plots across all `*_newlayout` model roots.

This stage is repo-wide (global), unlike Stage-3 which processes a specific model root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from stage4.global_summaries import (
    write_behavioural_shift_reliability_table,
    write_global_csvs,
    write_global_plots,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-4: build global significance CSVs/plots across *_newlayout.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        default="global_summaries",
        help="Directory for output CSVs/plots (default: global_summaries).",
    )
    parser.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to write global CSVs (default: enabled). Use --no-summarize to skip.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to generate plots (heatmap + effect plot per model_root).",
    )
    parser.add_argument(
        "--reliability-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to write behavioural-shift reliability binomial p-value tables (default: enabled).",
    )
    parser.add_argument(
        "--effect-metric",
        default="effect_cramers_v",
        choices=["effect_cramers_v", "effect_tv", "effect_js"],
        help="Effect size column to plot (when --plot).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    print(f"[Stage-4/global] Args: {args}")

    if args.summarize:
        long_path, summary_path, n_rows, n_groups = write_global_csvs(root, out_dir)
        print(f"[Stage-4/global] Wrote {long_path} ({n_rows} rows)")
        print(f"[Stage-4/global] Wrote {summary_path} ({n_groups} groups)")

    if args.reliability_table:
        csv_path, md_path, tex_path = write_behavioural_shift_reliability_table(root, out_dir)
        print(f"[Stage-4/global] Wrote {csv_path}")
        print(f"[Stage-4/global] Wrote {md_path}")
        print(f"[Stage-4/global] Wrote {tex_path}")

    if args.plot:
        written = write_global_plots(root, out_dir, effect_metric=args.effect_metric)
        for model_root, heatmap_path, effect_path in written:
            tag = model_root or "(unknown_model_root)"
            print(f"[Stage-4/global] Wrote {tag}: {heatmap_path}")
            print(f"[Stage-4/global] Wrote {tag}: {effect_path}")


if __name__ == "__main__":
    main()
