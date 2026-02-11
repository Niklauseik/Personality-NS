﻿# 分布偏移（Shift）说明与明细表（newlayout）

> 本文专门说明并汇总“分布偏移/方向性偏移（shift）”相关结果与明细表，配合 `EFFECT_EXPLANATION.md` 与 `SIGNIFICANCE_EXPLANATION.md` 使用。  
> 数据来源：`*_newlayout/*/summaries/sentiment.csv`（使用 `run=avg`）。

---

## 1) 我们如何定义“偏移方向”

为了跨数据集统一刻画“更偏正 / 更偏负”，对每个数据集定义一个情绪轴标量 $s$：

- 对情绪分类（positive/negative[/neutral]，覆盖 imdb / imdb_sklearn / sst2 / fiqasa）：  
  $$
  s = p(\text{positive}) - p(\text{negative})
  $$
- 对金融（bullish/bearish/neutral，news）：  
  $$
  s = p(\text{bullish}) - p(\text{bearish})
  $$
- 对心理健康（normal/depression，mental）：  
  $$
  s = p(\text{normal}) - p(\text{depression})
  $$

对相对性格模型与 base 的差异，用：

$$
\Delta s = s_{tuned} - s_{base}
$$

解释：

- $\Delta s > 0$：相对 base 更“正向”（更 positive / 更 bullish / 更 normal）
- $\Delta s < 0$：相对 base 更“负向”（更 negative / 更 bearish / 更 depression）

重要说明：

- 我们**按数据集分别统计**，不跨数据集合并“负向”语义（例如 mental 的 depression 与电影情绪的 negative 不应当强行视为同一语义标签）。

---

## 2) 什么叫“反向偏移”

对每个维度的两端性格（energy: E/I；information: N/S；decision: F/T；execution: J/P），如果在同一数据集上：

$$
\Delta s(\text{左端}) \cdot \Delta s(\text{右端}) < 0
$$

则称为**反向偏移**（推-拉式拉开）：一个更正向、另一个更负向。

---

## 3) 总表：按数据集拆分的“反向偏移”统计

表格单元格含义：在该（数据集 × 维度）下，3 个模型（llama-3b / qwen-3b / qwen-7b）中，有多少个满足“反向偏移”。

| 数据集 | energy | information | decision | execution |
|---|---:|---:|---:|---:|
| fiqasa | 3/3 (100.0%) | 2/3 (66.7%) | 3/3 (100.0%) | 2/3 (66.7%) |
| imdb | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 2/3 (66.7%) |
| imdb_sklearn | 2/3 (66.7%) | 1/3 (33.3%) | 3/3 (100.0%) | 1/3 (33.3%) |
| mental | 1/3 (33.3%) | 1/3 (33.3%) | 2/3 (66.7%) | 0/3 (0.0%) |
| news | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/3 (0.0%) |
| sst2 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| **总计** | **15/18 (83.3%)** | **13/18 (72.2%)** | **17/18 (94.4%)** | **8/18 (44.4%)** |

---

## 4) 明细 1：非反向偏移清单（逐模型 × 维度 × 数据集）

下表列出“不是反向偏移”的所有情况，并给出两个子模型相对 base 的 $\Delta s$（同号意味着同向偏移）。

| 模型 | 维度 | 数据集 | 性格1 | $\Delta s$(性格1) | 性格2 | $\Delta s$(性格2) | 现象 |
|---|---|---|---|---:|---|---:|---|
| qwen-3b | information | fiqasa | N | +0.0205 | S | +0.0009 | 同号(++) |
| llama-3b | execution | fiqasa | J | +0.1211 | P | +0.1032 | 同号(++) |
| qwen-3b | execution | imdb | J | +0.0020 | P | +0.0030 | 同号(++) |
| qwen-7b | energy | imdb_sklearn | E | +0.0227 | I | +0.0045 | 同号(++) |
| qwen-3b | information | imdb_sklearn | N | +0.0353 | S | +0.0193 | 同号(++) |
| qwen-7b | information | imdb_sklearn | N | +0.0217 | S | +0.0061 | 同号(++) |
| qwen-3b | execution | imdb_sklearn | J | +0.0282 | P | +0.0262 | 同号(++) |
| qwen-7b | execution | imdb_sklearn | J | +0.0104 | P | +0.0171 | 同号(++) |
| qwen-3b | energy | mental | E | -0.1010 | I | -0.1261 | 同号(--) |
| qwen-7b | energy | mental | E | -0.0844 | I | -0.1234 | 同号(--) |
| qwen-3b | information | mental | N | -0.2086 | S | -0.0330 | 同号(--) |
| qwen-7b | information | mental | N | -0.1664 | S | -0.0449 | 同号(--) |
| llama-3b | decision | mental | F | -0.0908 | T | -0.0114 | 同号(--) |
| llama-3b | execution | mental | J | +0.0980 | P | +0.7119 | 同号(++) |
| qwen-3b | execution | mental | J | -0.1233 | P | -0.1094 | 同号(--) |
| qwen-7b | execution | mental | J | -0.1168 | P | -0.0811 | 同号(--) |
| llama-3b | execution | news | J | -0.0671 | P | -0.0208 | 同号(--) |
| qwen-3b | execution | news | J | -0.0023 | P | -0.0173 | 同号(--) |
| qwen-7b | execution | news | J | -0.0015 | P | -0.0164 | 同号(--) |

---

## 5) 明细 2：每个（模型 × 维度）在多少个数据集上呈现反向偏移

| 模型 | 维度 | 反向偏移数据集数/总数据集数 | 反向偏移数据集 | 非反向数据集 |
|---|---|---:|---|---|
| llama-3b | energy | 6/6 | fiqasa, imdb, imdb_sklearn, mental, news, sst2 | - |
| llama-3b | information | 6/6 | fiqasa, imdb, imdb_sklearn, mental, news, sst2 | - |
| llama-3b | decision | 5/6 | fiqasa, imdb, imdb_sklearn, news, sst2 | mental |
| llama-3b | execution | 3/6 | imdb, imdb_sklearn, sst2 | fiqasa, mental, news |
| qwen-3b | energy | 5/6 | fiqasa, imdb, imdb_sklearn, news, sst2 | mental |
| qwen-3b | information | 3/6 | imdb, news, sst2 | fiqasa, imdb_sklearn, mental |
| qwen-3b | decision | 6/6 | fiqasa, imdb, imdb_sklearn, mental, news, sst2 | - |
| qwen-3b | execution | 2/6 | fiqasa, sst2 | imdb, imdb_sklearn, mental, news |
| qwen-7b | energy | 4/6 | fiqasa, imdb, news, sst2 | imdb_sklearn, mental |
| qwen-7b | information | 4/6 | fiqasa, imdb, news, sst2 | imdb_sklearn, mental |
| qwen-7b | decision | 6/6 | fiqasa, imdb, imdb_sklearn, mental, news, sst2 | - |
| qwen-7b | execution | 3/6 | fiqasa, imdb, sst2 | imdb_sklearn, mental, news |

---

## 6) 补充：同领域（电影域）跨数据集一致性（方向符号是否一致）

对每个（模型家族 × 维度 × 子模型）组合，检验其在 imdb/imdb_sklearn/sst2 上的 $\Delta s$ 符号是否一致：

| 维度 | 电影域符号一致（/总数） | 一致率 |
|---|---:|---:|
| energy | 5/6 | 83.3% |
| information | 3/6 | 50.0% |
| decision | 6/6 | 100.0% |
| execution | 4/6 | 66.7% |

直观示例（单元格为 “$\Delta s$(第一子模型) / $\Delta s$(第二子模型)”）：

| 模型 | decision: imdb (F/T) | decision: imdb_sklearn (F/T) | decision: sst2 (F/T) |
|---|---:|---:|---:|
| llama-3b | +0.0416 / -0.0693 | +0.0278 / -0.0915 | +0.1409 / -0.1668 |
| qwen-3b | +0.0320 / -0.0252 | +0.0202 / -0.0221 | +0.1131 / -0.1062 |
| qwen-7b | +0.0181 / -0.0263 | +0.0184 / -0.0274 | +0.0503 / -0.0576 |

| 模型 | energy: imdb (E/I) | energy: imdb_sklearn (E/I) | energy: sst2 (E/I) |
|---|---:|---:|---:|
| llama-3b | +0.0236 / -0.0228 | +0.0220 / -0.0122 | +0.0933 / -0.0606 |
| qwen-3b | +0.0244 / -0.0276 | +0.0504 / -0.0014 | +0.0896 / -0.0933 |
| qwen-7b | +0.0107 / -0.0133 | +0.0227 / +0.0045 | +0.0252 / -0.0254 |
