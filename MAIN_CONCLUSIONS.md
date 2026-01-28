﻿# Effect 偏移分析（newlayout）

本文在 `EFFECT_REPORT.md` 的结果基础上给出解释性分析。所有数值均为跨 runs 的均值。
训练数据规模（每个维度、每个类别）：
decision=12159，energy=2050，execution=7378，information=23233。

## 1) 维度总体概览（跨模型 + 数据集均值）

| 维度 | 训练样本/类 | CramersV 均值 | TV 均值 | JS 均值 |
|---|---:|---:|---:|---:|
| ST-NF | - | 0.254245 | 0.139490 | 0.028337 |
| decision | 12159 | 0.149911 | 0.046640 | 0.003615 |
| energy | 2050 | 0.137494 | 0.044843 | 0.005835 |
| execution | 7378 | 0.128705 | 0.043964 | 0.007173 |
| information | 23233 | 0.123151 | 0.039741 | 0.005066 |

## 2) 模型总体概览（跨维度 + 数据集均值）

| 模型 | CramersV 均值 | TV 均值 | JS 均值 |
|---|---:|---:|---:|
| llama-3b_newlayout | 0.190670 | 0.085946 | 0.016402 |
| qwen-3b_newlayout | 0.125372 | 0.035323 | 0.002012 |
| qwen-7b_newlayout | 0.104297 | 0.023508 | 0.000836 |

## 3) 数据集敏感度（跨模型 + 维度均值）

| 数据集 | CramersV 均值 | TV 均值 | JS 均值 |
|---|---:|---:|---:|
| fiqasa | 0.192251 | 0.089367 | 0.019318 |
| imdb | 0.075090 | 0.014813 | 0.000841 |
| imdb_sklearn | 0.077578 | 0.016906 | 0.000882 |
| mental | 0.211644 | 0.066030 | 0.009295 |
| news | 0.166411 | 0.071153 | 0.007738 |
| sst2 | 0.141038 | 0.048679 | 0.005034 |

## 4) 模型 × 维度（跨数据集均值）

| 模型 | 维度 | CramersV 均值 | TV 均值 | JS 均值 |
|---|---|---:|---:|---:|
| llama-3b_newlayout | ST-NF | 0.254245 | 0.139490 | 0.028337 |
| llama-3b_newlayout | decision | 0.170735 | 0.066010 | 0.007063 |
| llama-3b_newlayout | energy | 0.168499 | 0.070180 | 0.014548 |
| llama-3b_newlayout | execution | 0.176664 | 0.079268 | 0.018905 |
| llama-3b_newlayout | information | 0.183210 | 0.074783 | 0.013156 |
| qwen-3b_newlayout | decision | 0.150731 | 0.043440 | 0.002702 |
| qwen-3b_newlayout | energy | 0.136753 | 0.040723 | 0.002181 |
| qwen-3b_newlayout | execution | 0.115057 | 0.032080 | 0.001930 |
| qwen-3b_newlayout | information | 0.098948 | 0.025048 | 0.001236 |
| qwen-7b_newlayout | decision | 0.128266 | 0.030468 | 0.001080 |
| qwen-7b_newlayout | energy | 0.107231 | 0.023626 | 0.000775 |
| qwen-7b_newlayout | execution | 0.094393 | 0.020544 | 0.000682 |
| qwen-7b_newlayout | information | 0.087295 | 0.019392 | 0.000804 |

## 5) 解释性结论与假设检验

- 数据集敏感度（按 CramersV 均值）：最高为 `mental`，最低为 `imdb`。可视为“更敏感/更容易产生偏移”的数据集。
- 训练规模 vs 偏移：将“维度均值”与训练样本规模对照。一般而言，训练样本越多，偏移越容易形成；但实际偏移大小还受任务匹配度和标签噪声影响。
- 维度排序（按 CramersV 均值从高到低）：ST-NF、decision、energy、execution、information。
- 模型规模排序（按 CramersV 均值从高到低）：llama-3b_newlayout、qwen-3b_newlayout、qwen-7b_newlayout。若更大模型在同样数据量下偏移更小，说明“数据量”可能是瓶颈。
- 小样本维度（energy=2050/类）理论上更难出现大偏移，需结合第 1 表进行验证。
- 大样本维度（information=23233/类）如果与情感任务更匹配，偏移应更明显；可与数据集敏感度表交叉验证偏移集中在哪些数据集。