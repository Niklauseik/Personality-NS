#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/autodl-tmp/personality:${PYTHONPATH:-}

cd /root/autodl-tmp/personality
mkdir -p logs dpo_outputs results-decision-rerun-20260420

echo "[START] $(date)"

/root/miniconda3/bin/python -m stage1.dpo_training_chat \
  --dimension decision \
  --preferred feeling \
  --base-model-path ./llama-3B-Instruct \
  --save-path ./dpo_outputs/llama-3B-Instruct_F_seed42_20260420

echo "[F_DONE] $(date)"

/root/miniconda3/bin/python -m stage1.dpo_training_chat \
  --dimension decision \
  --preferred thinking \
  --base-model-path ./llama-3B-Instruct \
  --save-path ./dpo_outputs/llama-3B-Instruct_T_seed42_20260420

echo "[T_DONE] $(date)"

/root/miniconda3/bin/python - <<'PY'
from stage1.run_benchmark import run_benchmarks

specs = [
    {"display_name": "BASE", "checkpoint_path": "./llama-3B-Instruct"},
    {"display_name": "F", "checkpoint_path": "./dpo_outputs/llama-3B-Instruct_F_seed42_20260420"},
    {"display_name": "T", "checkpoint_path": "./dpo_outputs/llama-3B-Instruct_T_seed42_20260420"},
]

run_benchmarks(specs, results_root="results-decision-rerun-20260420")
PY

echo "[BENCH_DONE] $(date)"
