# Benchmark Capability Changes By Dataset

Values are reported as tuned minus base. Each row is one model family, one MBTI pole, and one benchmark dataset.

Notes:
- Metrics were recomputed with the local benchmark evaluator in [stage2/evaluate_benchmarks.py](C:/Users/BiaoPuYun/Personality-NS/stage2/evaluate_benchmarks.py).
- `GSM8K` only reports accuracy, so all `F1` cells for `GSM8K` are `NA`.
- `Llama-3.2-3B / Decision` uses the accepted summary at [cluster_results/llama-3b/decision/first_run/benchmark/benchmark_metrics_summary.txt](C:/Users/BiaoPuYun/Personality-NS/cluster_results/llama-3b/decision/first_run/benchmark/benchmark_metrics_summary.txt), because the local raw `F/T` benchmark folders were later overwritten by another rerun.
- `Qwen2.5-3B / Energy / E,I` benchmark raw files are not present locally under `qwen-3b_newlayout/energy/*/benchmark`, so those rows are marked `NA`.
- `Llama-3.2-3B` uses dimension-specific accepted base sources: energy base rerun `2026-04-03`, information base rerun `2026-04-03`, execution base rerun `2026-04-04`, and decision accepted summary source above.
- `Qwen2.5-7B / Execution / J,P` uses the paired base benchmark from the remote `results-J-P` rerun downloaded on `2026-05-02`.

| Model | Dimension | Type | Dataset | Base Acc | Tuned Acc | Delta Acc | Base F1 | Tuned F1 | Delta F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-3B | Energy | E | ARC (easy) | 0.8615 | 0.8594 | -0.0021 | 0.8610 | 0.8589 | -0.0021 |
| Llama-3.2-3B | Energy | E | BoolQ | 0.7320 | 0.7490 | +0.0170 | 0.7318 | 0.7479 | +0.0161 |
| Llama-3.2-3B | Energy | E | GSM8K | 0.0731 | 0.0731 | +0.0000 | NA | NA | NA |
| Llama-3.2-3B | Energy | I | ARC (easy) | 0.8615 | 0.8651 | +0.0036 | 0.8610 | 0.8647 | +0.0037 |
| Llama-3.2-3B | Energy | I | BoolQ | 0.7320 | 0.7055 | -0.0265 | 0.7318 | 0.7055 | -0.0263 |
| Llama-3.2-3B | Energy | I | GSM8K | 0.0731 | 0.0677 | -0.0054 | NA | NA | NA |
| Llama-3.2-3B | Information | N | ARC (easy) | 0.8615 | 0.8557 | -0.0058 | 0.8610 | 0.8550 | -0.0060 |
| Llama-3.2-3B | Information | N | BoolQ | 0.7320 | 0.7480 | +0.0160 | 0.7318 | 0.7462 | +0.0144 |
| Llama-3.2-3B | Information | N | GSM8K | 0.0831 | 0.0586 | -0.0245 | NA | NA | NA |
| Llama-3.2-3B | Information | S | ARC (easy) | 0.8615 | 0.8400 | -0.0215 | 0.8610 | 0.8405 | -0.0205 |
| Llama-3.2-3B | Information | S | BoolQ | 0.7320 | 0.7270 | -0.0050 | 0.7318 | 0.7269 | -0.0049 |
| Llama-3.2-3B | Information | S | GSM8K | 0.0831 | 0.0924 | +0.0093 | NA | NA | NA |
| Llama-3.2-3B | Decision | F | ARC (easy) | 0.7034 | 0.7367 | +0.0333 | 0.7094 | 0.7395 | +0.0301 |
| Llama-3.2-3B | Decision | F | BoolQ | 0.7330 | 0.7335 | +0.0005 | 0.7327 | 0.7330 | +0.0003 |
| Llama-3.2-3B | Decision | F | GSM8K | 0.0415 | 0.0400 | -0.0015 | NA | NA | NA |
| Llama-3.2-3B | Decision | T | ARC (easy) | 0.7034 | 0.7493 | +0.0459 | 0.7094 | 0.7536 | +0.0442 |
| Llama-3.2-3B | Decision | T | BoolQ | 0.7330 | 0.7010 | -0.0320 | 0.7327 | 0.7009 | -0.0318 |
| Llama-3.2-3B | Decision | T | GSM8K | 0.0415 | 0.0531 | +0.0116 | NA | NA | NA |
| Llama-3.2-3B | Execution | J | ARC (easy) | 0.8620 | 0.8604 | -0.0016 | 0.8616 | 0.8600 | -0.0016 |
| Llama-3.2-3B | Execution | J | BoolQ | 0.7320 | 0.7470 | +0.0150 | 0.7318 | 0.7464 | +0.0146 |
| Llama-3.2-3B | Execution | J | GSM8K | 0.0824 | 0.0808 | -0.0016 | NA | NA | NA |
| Llama-3.2-3B | Execution | P | ARC (easy) | 0.8620 | 0.8489 | -0.0131 | 0.8616 | 0.8486 | -0.0130 |
| Llama-3.2-3B | Execution | P | BoolQ | 0.7320 | 0.7050 | -0.0270 | 0.7318 | 0.7050 | -0.0268 |
| Llama-3.2-3B | Execution | P | GSM8K | 0.0824 | 0.0565 | -0.0259 | NA | NA | NA |
| Qwen2.5-3B | Energy | E | ARC (easy) | 0.9324 | NA | NA | 0.9324 | NA | NA |
| Qwen2.5-3B | Energy | E | BoolQ | 0.7940 | NA | NA | 0.7925 | NA | NA |
| Qwen2.5-3B | Energy | E | GSM8K | 0.0686 | NA | NA | NA | NA | NA |
| Qwen2.5-3B | Energy | I | ARC (easy) | 0.9324 | NA | NA | 0.9324 | NA | NA |
| Qwen2.5-3B | Energy | I | BoolQ | 0.7940 | NA | NA | 0.7925 | NA | NA |
| Qwen2.5-3B | Energy | I | GSM8K | 0.0686 | NA | NA | NA | NA | NA |
| Qwen2.5-3B | Information | N | ARC (easy) | 0.9324 | 0.9370 | +0.0046 | 0.9324 | 0.9370 | +0.0046 |
| Qwen2.5-3B | Information | N | BoolQ | 0.7940 | 0.8009 | +0.0069 | 0.7925 | 0.7991 | +0.0066 |
| Qwen2.5-3B | Information | N | GSM8K | 0.0686 | 0.0662 | -0.0024 | NA | NA | NA |
| Qwen2.5-3B | Information | S | ARC (easy) | 0.9324 | 0.9327 | +0.0003 | 0.9324 | 0.9327 | +0.0003 |
| Qwen2.5-3B | Information | S | BoolQ | 0.7940 | 0.7913 | -0.0027 | 0.7925 | 0.7901 | -0.0024 |
| Qwen2.5-3B | Information | S | GSM8K | 0.0686 | 0.0785 | +0.0099 | NA | NA | NA |
| Qwen2.5-3B | Decision | F | ARC (easy) | 0.9324 | 0.9309 | -0.0015 | 0.9324 | 0.9309 | -0.0015 |
| Qwen2.5-3B | Decision | F | BoolQ | 0.7940 | 0.7990 | +0.0050 | 0.7925 | 0.7972 | +0.0047 |
| Qwen2.5-3B | Decision | F | GSM8K | 0.0686 | 0.0717 | +0.0031 | NA | NA | NA |
| Qwen2.5-3B | Decision | T | ARC (easy) | 0.9324 | 0.9305 | -0.0019 | 0.9324 | 0.9304 | -0.0020 |
| Qwen2.5-3B | Decision | T | BoolQ | 0.7940 | 0.7920 | -0.0020 | 0.7925 | 0.7906 | -0.0019 |
| Qwen2.5-3B | Decision | T | GSM8K | 0.0686 | 0.0716 | +0.0030 | NA | NA | NA |
| Qwen2.5-3B | Execution | J | ARC (easy) | 0.9324 | 0.9335 | +0.0011 | 0.9324 | 0.9335 | +0.0011 |
| Qwen2.5-3B | Execution | J | BoolQ | 0.7940 | 0.8053 | +0.0113 | 0.7925 | 0.8034 | +0.0109 |
| Qwen2.5-3B | Execution | J | GSM8K | 0.0686 | 0.0808 | +0.0122 | NA | NA | NA |
| Qwen2.5-3B | Execution | P | ARC (easy) | 0.9324 | 0.9329 | +0.0005 | 0.9324 | 0.9329 | +0.0005 |
| Qwen2.5-3B | Execution | P | BoolQ | 0.7940 | 0.7864 | -0.0076 | 0.7925 | 0.7852 | -0.0073 |
| Qwen2.5-3B | Execution | P | GSM8K | 0.0686 | 0.0654 | -0.0032 | NA | NA | NA |
| Qwen2.5-7B | Energy | E | ARC (easy) | 0.9579 | 0.9589 | +0.0010 | 0.9579 | 0.9589 | +0.0010 |
| Qwen2.5-7B | Energy | E | BoolQ | 0.8702 | 0.8687 | -0.0015 | 0.8662 | 0.8649 | -0.0013 |
| Qwen2.5-7B | Energy | E | GSM8K | 0.0923 | 0.1108 | +0.0185 | NA | NA | NA |
| Qwen2.5-7B | Energy | I | ARC (easy) | 0.9579 | 0.9588 | +0.0009 | 0.9579 | 0.9588 | +0.0009 |
| Qwen2.5-7B | Energy | I | BoolQ | 0.8702 | 0.8645 | -0.0057 | 0.8662 | 0.8610 | -0.0052 |
| Qwen2.5-7B | Energy | I | GSM8K | 0.0923 | 0.1092 | +0.0169 | NA | NA | NA |
| Qwen2.5-7B | Information | N | ARC (easy) | 0.9579 | 0.9522 | -0.0057 | 0.9579 | 0.9521 | -0.0058 |
| Qwen2.5-7B | Information | N | BoolQ | 0.8702 | 0.8699 | -0.0003 | 0.8662 | 0.8657 | -0.0005 |
| Qwen2.5-7B | Information | N | GSM8K | 0.0923 | 0.1031 | +0.0108 | NA | NA | NA |
| Qwen2.5-7B | Information | S | ARC (easy) | 0.9579 | 0.9574 | -0.0005 | 0.9579 | 0.9575 | -0.0004 |
| Qwen2.5-7B | Information | S | BoolQ | 0.8702 | 0.8653 | -0.0049 | 0.8662 | 0.8620 | -0.0042 |
| Qwen2.5-7B | Information | S | GSM8K | 0.0923 | 0.1092 | +0.0169 | NA | NA | NA |
| Qwen2.5-7B | Decision | F | ARC (easy) | 0.9579 | 0.9574 | -0.0005 | 0.9579 | 0.9575 | -0.0004 |
| Qwen2.5-7B | Decision | F | BoolQ | 0.8702 | 0.8730 | +0.0028 | 0.8662 | 0.8687 | +0.0025 |
| Qwen2.5-7B | Decision | F | GSM8K | 0.0923 | 0.0939 | +0.0016 | NA | NA | NA |
| Qwen2.5-7B | Decision | T | ARC (easy) | 0.9579 | 0.9580 | +0.0001 | 0.9579 | 0.9580 | +0.0001 |
| Qwen2.5-7B | Decision | T | BoolQ | 0.8702 | 0.8683 | -0.0019 | 0.8662 | 0.8646 | -0.0016 |
| Qwen2.5-7B | Decision | T | GSM8K | 0.0923 | 0.0938 | +0.0015 | NA | NA | NA |
| Qwen2.5-7B | Execution | J | ARC (easy) | 0.9556 | 0.9556 | +0.0000 | 0.9556 | 0.9556 | +0.0000 |
| Qwen2.5-7B | Execution | J | BoolQ | 0.8661 | 0.8688 | +0.0026 | 0.8624 | 0.8650 | +0.0026 |
| Qwen2.5-7B | Execution | J | GSM8K | 0.1577 | 0.1515 | -0.0062 | NA | NA | NA |
| Qwen2.5-7B | Execution | P | ARC (easy) | 0.9556 | 0.9555 | -0.0001 | 0.9556 | 0.9555 | -0.0001 |
| Qwen2.5-7B | Execution | P | BoolQ | 0.8661 | 0.8652 | -0.0010 | 0.8624 | 0.8615 | -0.0009 |
| Qwen2.5-7B | Execution | P | GSM8K | 0.1577 | 0.1546 | -0.0031 | NA | NA | NA |
