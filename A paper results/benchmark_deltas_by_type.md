# Benchmark Deltas By Type

Values are reported as tuned minus base.

Update note:
On 2026-04-03, the Llama-3.2-3B Energy and Information benchmark rows were replaced with greedy-decoding reruns from the IE and NS AutoDL machines. Earlier sampling-based values were discarded.
On 2026-04-04, the Qwen2.5-7B Execution rows were filled with the newly downloaded J/P benchmark reruns.
On 2026-04-04, the Llama-3.2-3B Execution rows were replaced with a same-batch base/J/P rerun from the new AutoDL machine.
On 2026-04-21, the Llama-3.2-3B Decision rows were replaced with the latest benchmark summary provided for the same base/F/T comparison. The earlier 2026-04-07 and same-day interim rerun decision numbers were discarded.

Archive:
Raw downloaded archives and rerun base-model CSV files are stored under `A paper results/archive/`.

| Model        | Dimension   | Type | ARC-Easy DeltaAcc | ARC-Easy DeltaF1 | BoolQ DeltaAcc | BoolQ DeltaF1 | GSM8K DeltaAcc |
| ------------ | ----------- | ---- | ----------------: | ---------------: | -------------: | ------------: | --------------: |
| Llama-3.2-3B | Energy      | E    |           -0.0021 |          -0.0021 |         +0.0170 |        +0.0161 |         +0.0000 |
|              |             | I    |           +0.0036 |          +0.0037 |         -0.0265 |        -0.0263 |         -0.0054 |
|              | Information | N    |           -0.0058 |          -0.0060 |         +0.0160 |        +0.0144 |         -0.0245 |
|              |             | S    |           -0.0215 |          -0.0205 |         -0.0050 |        -0.0049 |         +0.0093 |
|              | Decision    | F    |           +0.0333 |          +0.0301 |         +0.0005 |        +0.0003 |         -0.0015 |
|              |             | T    |           +0.0459 |          +0.0442 |         -0.0320 |        -0.0318 |         +0.0116 |
|              | Execution   | J    |           -0.0016 |          -0.0016 |         +0.0150 |        +0.0146 |         -0.0016 |
|              |             | P    |           -0.0131 |          -0.0130 |         -0.0270 |        -0.0268 |         -0.0259 |
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
|              | Execution   | J    |           -0.0023 |          -0.0023 |         -0.0014 |        -0.0012 |         +0.0592 |
|              |             | P    |           -0.0024 |          -0.0024 |         -0.0050 |        -0.0047 |         +0.0623 |
