# -*- coding: utf-8 -*-
"""Stage-3 entrypoint for benchmark capability drop t-tests."""

from __future__ import annotations

from typing import Sequence

from stage3.benchmark_drop_ttest import build_arg_parser, run


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(f"[Stage-3/benchmark-drop] Args: {args}")
    paths = run(args)
    for name, path in paths.items():
        print(f"[Stage-3/benchmark-drop] Wrote {name}: {path}")


if __name__ == "__main__":
    main()
