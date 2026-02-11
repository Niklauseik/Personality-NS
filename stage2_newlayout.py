# -*- coding: utf-8 -*-
"""Stage-2 new-layout entrypoint (CLI).

Convenience wrapper for processing one or more new-layout model roots (contain `base/`).
Internally forwards to Stage-2 pipeline in `stage2/process_results.py`.
"""

from __future__ import annotations

import argparse
from typing import Sequence
from types import SimpleNamespace

from stage2.process_results import run as run_stage2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 (new layout): process one or more model roots (contain base/).")
    parser.add_argument(
        "--model-root",
        nargs="+",
        default=None,
        help="One or more new-layout model roots (contain base/).",
    )
    parser.add_argument(
        "--model-glob",
        action="append",
        default=[],
        help='Glob pattern(s) (relative to CWD) to discover model roots, e.g. "*_newlayout". Can be repeated.',
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining model roots if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list which model roots would be processed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model_roots = args.model_root or []
    print(f"[Stage-2/newlayout] Args: {args}")
    forwarded = SimpleNamespace(
        results_root=model_roots,
        results_glob=args.model_glob,
        continue_on_error=bool(args.continue_on_error),
        dry_run=bool(args.dry_run),
    )
    run_stage2(forwarded)


if __name__ == "__main__":
    main()
