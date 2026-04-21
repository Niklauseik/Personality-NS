# Table 2. Opposite-Direction Shift Across Datasets And Model Families

`✓` indicates an opposite-direction shift between the two poles of one MBTI dimension relative to the base model on that dataset, computed from the `avg` rows in each dimension's `summaries/sentiment.csv`.
This table was recomputed directly from the real result files under `llama-3b_newlayout/`, `qwen-3b_newlayout/`, and `qwen-7b_newlayout/` using `Δs = s_tuned - s_base`, where `s = positive - negative`, `normal - depression`, or `bullish - bearish` depending on the dataset.

| Dimension | Dataset | Llama-3.2-3B | Qwen2.5-3B | Qwen2.5-7B | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Energy | FiQA-SA | ✓ | ✓ | ✓ | 3/3 |
|  | IMDb | ✓ | ✓ | ✓ | 3/3 |
|  | IMDb-Sklearn | ✓ | ✓ | ✗ | 2/3 |
|  | Mental | ✓ | ✗ | ✗ | 1/3 |
|  | News | ✓ | ✓ | ✓ | 3/3 |
|  | SST-2 | ✓ | ✓ | ✓ | 3/3 |
|  | Total | 6/6 | 5/6 | 4/6 | 15/18 |
| Information | FiQA-SA | ✓ | ✗ | ✓ | 2/3 |
|  | IMDb | ✓ | ✓ | ✓ | 3/3 |
|  | IMDb-Sklearn | ✓ | ✗ | ✗ | 1/3 |
|  | Mental | ✓ | ✗ | ✗ | 1/3 |
|  | News | ✓ | ✓ | ✓ | 3/3 |
|  | SST-2 | ✓ | ✓ | ✓ | 3/3 |
|  | Total | 6/6 | 3/6 | 4/6 | 13/18 |
| Decision | FiQA-SA | ✓ | ✓ | ✓ | 3/3 |
|  | IMDb | ✓ | ✓ | ✓ | 3/3 |
|  | IMDb-Sklearn | ✓ | ✓ | ✓ | 3/3 |
|  | Mental | ✗ | ✓ | ✓ | 2/3 |
|  | News | ✓ | ✓ | ✓ | 3/3 |
|  | SST-2 | ✓ | ✓ | ✓ | 3/3 |
|  | Total | 5/6 | 6/6 | 6/6 | 17/18 |
| Execution | FiQA-SA | ✗ | ✓ | ✓ | 2/3 |
|  | IMDb | ✓ | ✗ | ✓ | 2/3 |
|  | IMDb-Sklearn | ✓ | ✗ | ✗ | 1/3 |
|  | Mental | ✗ | ✗ | ✗ | 0/3 |
|  | News | ✗ | ✗ | ✗ | 0/3 |
|  | SST-2 | ✓ | ✓ | ✓ | 3/3 |
|  | Total | 3/6 | 2/6 | 3/6 | 8/18 |
