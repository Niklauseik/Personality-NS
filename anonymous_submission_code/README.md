# Anonymous Code Package

This package contains the core implementation needed to reproduce the main experimental workflow.
It intentionally excludes datasets, generated results, plots, slide decks, local logs, remote-machine commands, credentials, and version-control history.

## Contents

```text
MBTI/
  code/               MBTI validation utilities
common/               Shared pipeline helpers
model_training/       DPO/LoRA training plus benchmark and sentiment inference
result_processing/    Invalid-output handling, metric evaluation, MBTI validation, and plots
statistical_analysis/ Significance summaries, plots, and benchmark drop tests
```

The compatibility entrypoints are:

```text
run_model_training.py
process_results.py
run_statistical_analysis.py
run_benchmark_drop_test.py
```

## Environment

Install the Python dependencies listed in `requirements.txt`. Training and local inference require GPU resources appropriate for the selected base model.

The optional invalid-label correction step in `result_processing/sentiment_label_correct.py` uses the OpenAI Python SDK. If no API key is configured in the runtime environment, the main result-processing pipeline skips that correction step and continues with the deterministic processing steps.

Datasets are not included in this anonymous code package. Place the required training, benchmark, sentiment, and MBTI validation files under the expected runtime paths, or pass the corresponding CLI path arguments when running the pipeline.

## Example Workflow

Train personality-adapted models and run evaluations:

```bash
python run_model_training.py \
  --dimension information \
  --model-path <base-model-checkpoint> \
  --sentiment-runs 2
```

Process one or more raw result directories or new-layout model roots:

```bash
python process_results.py --results-root results-base-model-N-S-run1
```

Summarize significance results and draw plots:

```bash
python run_statistical_analysis.py --model-glob "*_newlayout"
```

Run the benchmark capability drop test on a benchmark-delta table:

```bash
python run_benchmark_drop_test.py \
  --input benchmark_deltas_by_dataset.md \
  --output-dir analysis_outputs/benchmark_drop_ttest
```

The benchmark drop-test input table should contain these columns:

```text
Model, Dimension, Type, Dataset, Delta Acc
```

## Notes

- Generated output directories are not included in this package.
- Dataset files are not included in this package.
- The code does not include trained checkpoints. Provide local or downloaded base-model/checkpoint paths at runtime.
- Runtime path placeholders should be replaced with locations in the review environment.
