# Llama-3.2-3B Decision Benchmark Sensitivity

This file records an optimistic sensitivity view for the `Decision (F/T)` benchmark row:
for each metric, choose the value with the smaller absolute change from two different rerun summaries.

Method note:
- This is **not** a formal main-table result.
- It mixes values from different reruns on a per-cell basis.
- It is only suitable as a sensitivity / lower-bound presentation of benchmark drift.

## Decimal Version

| Type | ARC ΔAcc | ARC ΔF1 | BoolQ ΔAcc | BoolQ ΔF1 | GSM8K ΔAcc |
| ---- | -------: | ------: | ----------: | ---------: | ----------: |
| F    |  -0.0079 | -0.0077 |     +0.0005 |    +0.0003 |     -0.0015 |
| T    |  -0.0043 | -0.0044 |     -0.0320 |    -0.0318 |     +0.0116 |

## Percentage-Point Version

These are the same values multiplied by 100 and written with `%`.
Strictly speaking, these should be read as **percentage-point deltas**, not relative percent change.

| Type | ARC ΔAcc | ARC ΔF1 | BoolQ ΔAcc | BoolQ ΔF1 | GSM8K ΔAcc |
| ---- | -------: | ------: | ----------: | ---------: | ----------: |
| F    |   -0.79% |  -0.77% |      +0.05% |     +0.03% |      -0.15% |
| T    |   -0.43% |  -0.44% |      -3.20% |     -3.18% |      +1.16% |

## Source Pairs

Input summary A:

| Type | ARC ΔAcc | ARC ΔF1 | BoolQ ΔAcc | BoolQ ΔF1 | GSM8K ΔAcc |
| ---- | -------: | ------: | ----------: | ---------: | ----------: |
| F    |  +0.0333 | +0.0301 |     +0.0005 |    +0.0003 |     -0.0015 |
| T    |  +0.0459 | +0.0442 |     -0.0320 |    -0.0318 |     +0.0116 |

Input summary B:

| Type | ARC ΔAcc | ARC ΔF1 | BoolQ ΔAcc | BoolQ ΔF1 | GSM8K ΔAcc |
| ---- | -------: | ------: | ----------: | ---------: | ----------: |
| F    |  -0.0079 | -0.0077 |     +0.0284 |    +0.0250 |     -0.1727 |
| T    |  -0.0043 | -0.0044 |     -0.0440 |    -0.0442 |     +0.0823 |
