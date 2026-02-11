# -*- coding: utf-8 -*-
"""Stage-5 helper: build global performance summaries/plots across all `*_newlayout` model roots (sentiment only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from stage5.global_performance import write_global_performance


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-5: global performance summaries (sentiment only).")
    parser.add_argument("--root", default=".", help="Repository root to scan (default: current directory).")
    parser.add_argument(
        "--output-dir",
        default="global_performance",
        help="Directory for output CSVs/plots (default: global_performance).",
    )
    parser.add_argument(
        "--run",
        default="avg",
        help="Which run to analyze (default: avg). e.g. run-001/run-002/avg.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Bucket threshold for 'effective' improve/decline (default: 0.005 = 0.5pp).",
    )
    parser.add_argument("--plot", action="store_true", help="Whether to generate plots (heatmaps + push-pull + domain bars).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    outputs = write_global_performance(
        root=root,
        output_dir=out_dir,
        run=str(args.run),
        threshold=float(args.threshold),
        plot=bool(args.plot),
    )
    print(f"[Stage-5/global] Wrote: {outputs.performance_long_csv}")
    print(f"[Stage-5/global] Wrote: {outputs.performance_pairwise_csv}")
    print(f"[Stage-5/global] Wrote: {outputs.performance_summary_csv}")
    if outputs.plots_dir is not None:
        print(f"[Stage-5/global] Wrote plots under: {outputs.plots_dir}")


if __name__ == "__main__":
    main()

