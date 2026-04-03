# Benchmark Deltas By Type

Values are reported as tuned minus base.

Update note:
On 2026-04-03, the Llama-3.2-3B Energy and Information benchmark rows were replaced with greedy-decoding reruns from the IE and NS AutoDL machines. Earlier sampling-based values were discarded.

Archive:
Raw downloaded archives and the rerun base-model CSV files are stored under `A paper results/archive/greedy_benchmark_raw_20260403/`.

| Model        | Dimension   | Type | ARC-Easy DeltaAcc | ARC-Easy DeltaF1 | BoolQ DeltaAcc | BoolQ DeltaF1 | GSM8K DeltaAcc |
| ------------ | ----------- | ---- | ----------------: | ---------------: | -------------: | ------------: | --------------: |
| Llama-3.2-3B | Energy      | E    |           -0.0021 |          -0.0021 |         +0.0170 |        +0.0161 |         +0.0000 |
|              |             | I    |           +0.0036 |          +0.0037 |         -0.0265 |        -0.0263 |         -0.0054 |
|              | Information | N    |           -0.0058 |          -0.0060 |         +0.0160 |        +0.0144 |         -0.0245 |
|              |             | S    |           -0.0215 |          -0.0205 |         -0.0050 |        -0.0049 |         +0.0093 |
|              | Decision    | F    |           +0.0787 |          +0.0724 |         +0.0095 |        +0.0091 |         -0.0353 |
|              |             | T    |           +0.0639 |          +0.0622 |         -0.0205 |        -0.0203 |         +0.0076 |
|              | Execution   | J    |           +0.1669 |          +0.1568 |         +0.0145 |        +0.0140 |         -0.0123 |
|              |             | P    |           -0.0509 |          -0.0499 |         -0.0175 |        -0.0173 |         -0.0054 |
| Qwen2.5-3B   | Energy      | E    |           -0.0013 |          -0.0013 |         +0.0115 |        +0.0110 |         +0.0031 |
|              |             | I    |           +0.0048 |          +0.0048 |         -0.0125 |        -0.0119 |         +0.0000 |
|              | Information | N    |           +0.0046 |          +0.0046 |         +0.0069 |        +0.0066 |         -0.0024 |
|              |             | S    |           +0.0003 |          +0.0002 |         -0.0027 |        -0.0023 |         +0.0099 |
|              | Decision    | F    |           -0.0015 |          -0.0015 |         +0.0050 |        +0.0047 |         +0.0031 |
|              |             | T    |           -0.0019 |          -0.0020 |         -0.0020 |        -0.0019 |         +0.0030 |
|              | Execution   | J    |           +0.0011 |          +0.0011 |         +0.0113 |        +0.0109 |         +0.0122 |
|              |             | P    |           +0.0006 |          +0.0005 |         -0.0076 |        -0.0073 |         -0.0032 |
| Qwen2.5-7B   | Energy      | E    |           +0.0010 |          +0.0010 |         -0.0015 |        -0.0013 |         +0.0185 |
|              |             | I    |           +0.0009 |          +0.0009 |         -0.0057 |        -0.0052 |         +0.0169 |
|              | Information | N    |           -0.0057 |          -0.0058 |         -0.0002 |        -0.0005 |         +0.0108 |
|              |             | S    |           -0.0004 |          -0.0004 |         -0.0048 |        -0.0042 |         +0.0169 |
|              | Decision    | F    |           -0.0005 |          -0.0005 |         +0.0028 |        +0.0025 |         +0.0016 |
|              |             | T    |           +0.0001 |          +0.0001 |         -0.0019 |        -0.0016 |         +0.0015 |
|              | Execution   | J    |                   |                  |                |               |                 |
|              |             | P    |                   |                  |                |               |                 |
