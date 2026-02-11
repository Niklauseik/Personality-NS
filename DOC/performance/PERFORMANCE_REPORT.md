# Performance Report（newlayout，sentiment only，run=avg）

> 本报告回答：性格微调后，相对 base 的 **Accuracy / Macro-F1** 是提高还是下降？  
> 统计口径见：`PERFORMANCE_EXPLANATION.md`。  
> 数据来源：`*_newlayout/*/summaries/sentiment.csv`（Stage-2 输出），Stage-5 汇总输出见 `global_performance/`。

默认设置：
- 只看 sentiment
- 使用 strict 指标：`accuracy_strict` + `f1_macro_strict`
- 使用 `run=avg`
- “有效提升/下降”阈值：`0.005`（0.5pp）

表格说明：
- `ΔAcc(X)` / `ΔF1(X)`：模型 X 相对 BASE 的差值（X - BASE）
- `winner_acc` / `winner_f1`：在同一数据集上，两端性格模型谁更好（使用同一阈值 0.005 做 tie 判定）

领域划分：
- movie：imdb / imdb_sklearn / sst2
- finance：fiqasa / news
- mental：mental

---

## 1) 核心结论（最强的“性格-领域”映射信号来自 decision）

### 1.1 decision（F/T）呈现最清晰的“电影域 vs 心理健康域”分化

在 `qwen-3b_newlayout` 与 `qwen-7b_newlayout` 上，decision 维度呈现非常稳定的模式：
- **movie：F 提升（或更接近提升），T 明显下降**
- **mental：T 提升，F 下降**

先看每个数据集的变化（run=avg）：

#### qwen-3b_newlayout / decision（逐数据集 Δ）
| dataset | domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0358 | -0.0546 | +0.0033 | -0.0287 | F | F |
| imdb | movie | +0.0044 | -0.0048 | +0.0045 | -0.0049 | F | F |
| imdb_sklearn | movie | +0.0057 | -0.0088 | +0.0058 | -0.0086 | F | F |
| mental | mental | -0.0286 | +0.0100 | -0.0287 | +0.0097 | T | T |
| news | finance | -0.0363 | +0.0134 | -0.0375 | +0.0076 | T | T |
| sst2 | movie | +0.0521 | -0.0501 | +0.0550 | -0.0558 | F | F |

#### qwen-7b_newlayout / decision（逐数据集 Δ）
| dataset | domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0350 | -0.0401 | +0.0205 | -0.0237 | F | F |
| imdb | movie | +0.0008 | -0.0041 | +0.0013 | -0.0044 | tie | F |
| imdb_sklearn | movie | +0.0005 | -0.0051 | +0.0017 | -0.0050 | F | F |
| mental | mental | -0.0308 | +0.0275 | -0.0328 | +0.0286 | T | T |
| news | finance | -0.0334 | +0.0091 | -0.0187 | -0.0040 | T | T |
| sst2 | movie | +0.0238 | -0.0314 | +0.0175 | -0.0240 | F | F |

#### llama-3b_newlayout / decision（逐数据集 Δ）
| dataset | domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.1381 | -0.0823 | +0.0956 | -0.0702 | F | F |
| imdb | movie | +0.0047 | -0.0297 | +0.0063 | -0.0259 | F | F |
| imdb_sklearn | movie | +0.0020 | -0.0303 | +0.0020 | -0.0308 | F | F |
| mental | mental | -0.0373 | -0.0004 | -0.0628 | -0.0041 | T | T |
| news | finance | -0.0558 | +0.0596 | -0.0493 | +0.0449 | T | T |
| sst2 | movie | +0.0505 | -0.1028 | +0.0477 | -0.0941 | F | F |

再按领域聚合（均值 Δ）：

#### qwen-3b_newlayout / decision（领域均值 Δ）
| domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) |
|---|---:|---:|---:|---:|
| movie | +0.0207 | -0.0212 | +0.0217 | -0.0231 |
| finance | -0.0002 | -0.0206 | -0.0171 | -0.0105 |
| mental | -0.0286 | +0.0100 | -0.0287 | +0.0097 |

#### qwen-7b_newlayout / decision（领域均值 Δ）
| domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) |
|---|---:|---:|---:|---:|
| movie | +0.0084 | -0.0135 | +0.0069 | -0.0111 |
| finance | +0.0008 | -0.0155 | +0.0009 | -0.0138 |
| mental | -0.0308 | +0.0275 | -0.0328 | +0.0286 |

> 解释：decision 维度同时在 Accuracy 与 Macro-F1 上一致，说明这不是“只靠偏向多数类撑起的假提升”，而更像真实的领域适配差异。

#### llama-3b_newlayout / decision（领域均值 Δ，趋势一致但 mental 更接近 tie）
| domain | ΔAcc(F) | ΔAcc(T) | ΔF1(F) | ΔF1(T) |
|---|---:|---:|---:|---:|
| movie | +0.0191 | -0.0543 | +0.0187 | -0.0503 |
| finance | +0.0411 | -0.0113 | +0.0231 | -0.0126 |
| mental | -0.0373 | -0.0004 | -0.0628 | -0.0041 |

---

## 2) 次强信号：energy（E/I）在电影域上稳定

energy 维度在三个模型家族上都表现为：
- **movie：E 相对 base 上升，I 相对 base 下降**

先看每个数据集的变化（run=avg）：

#### llama-3b_newlayout / energy（逐数据集 Δ）
| dataset | domain | ΔAcc(E) | ΔAcc(I) | ΔF1(E) | ΔF1(I) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.2438 | +0.0217 | +0.1353 | +0.0104 | E | E |
| imdb | movie | +0.0045 | -0.0082 | +0.0043 | -0.0077 | E | E |
| imdb_sklearn | movie | +0.0037 | -0.0035 | +0.0038 | -0.0035 | E | E |
| mental | mental | +0.0280 | -0.0319 | +0.0466 | -0.0498 | E | E |
| news | finance | -0.1002 | +0.0686 | -0.0839 | +0.0449 | I | I |
| sst2 | movie | +0.0540 | -0.0371 | +0.0413 | -0.0311 | E | E |

#### qwen-3b_newlayout / energy（逐数据集 Δ）
| dataset | domain | ΔAcc(E) | ΔAcc(I) | ΔF1(E) | ΔF1(I) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0375 | -0.0384 | +0.0171 | -0.0216 | E | E |
| imdb | movie | +0.0042 | -0.0068 | +0.0042 | -0.0069 | E | E |
| imdb_sklearn | movie | +0.0150 | -0.0011 | +0.0152 | -0.0011 | E | E |
| mental | mental | -0.0231 | -0.0331 | -0.0231 | -0.0333 | E | E |
| news | finance | +0.0013 | +0.0197 | -0.0002 | +0.0009 | I | tie |
| sst2 | movie | +0.0416 | -0.0437 | +0.0442 | -0.0486 | E | E |

#### qwen-7b_newlayout / energy（逐数据集 Δ）
| dataset | domain | ΔAcc(E) | ΔAcc(I) | ΔF1(E) | ΔF1(I) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0307 | -0.0264 | +0.0188 | -0.0149 | E | E |
| imdb | movie | +0.0012 | -0.0021 | +0.0010 | -0.0023 | tie | tie |
| imdb_sklearn | movie | +0.0028 | +0.0012 | +0.0044 | +0.0027 | tie | tie |
| mental | mental | -0.0332 | -0.0495 | -0.0350 | -0.0527 | E | E |
| news | finance | -0.0101 | +0.0141 | -0.0042 | +0.0111 | I | I |
| sst2 | movie | +0.0090 | -0.0114 | +0.0079 | -0.0103 | E | E |

再按领域聚合（均值 ΔAcc，movie 域上最稳定）：
- llama-3b_newlayout：movie `E +0.0208`，`I -0.0163`
- qwen-3b_newlayout：movie `E +0.0203`，`I -0.0172`
- qwen-7b_newlayout：movie `E +0.0043`，`I -0.0041`

mental/finance 上的模式更依赖模型家族：在 qwen 系列里，mental 领域两端都更容易下降，但 **E 通常“更不差”**（pairwise 对比更占优）。

---

## 3) information（N/S）与 execution（J/P）：更不稳定，需结合具体数据集解释

先报告每个数据集（run=avg），再讨论领域层面的规律。

### 3.1 information（N/S）：逐数据集 Δ

#### llama-3b_newlayout / information（逐数据集 Δ）
| dataset | domain | ΔAcc(N) | ΔAcc(S) | ΔF1(N) | ΔF1(S) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0375 | +0.0895 | +0.0256 | +0.0405 | S | S |
| imdb | movie | -0.0132 | -0.0142 | -0.0031 | -0.0156 | tie | N |
| imdb_sklearn | movie | +0.0006 | -0.0204 | +0.0007 | -0.0207 | N | N |
| mental | mental | -0.0340 | +0.0254 | -0.0594 | +0.0298 | S | S |
| news | finance | -0.1143 | +0.0113 | -0.0877 | +0.0030 | S | S |
| sst2 | movie | -0.0292 | -0.0483 | +0.0026 | -0.0728 | N | N |

#### qwen-3b_newlayout / information（逐数据集 Δ）
| dataset | domain | ΔAcc(N) | ΔAcc(S) | ΔF1(N) | ΔF1(S) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0026 | +0.0119 | +0.0035 | +0.0098 | S | S |
| imdb | movie | +0.0023 | -0.0038 | +0.0023 | -0.0038 | N | N |
| imdb_sklearn | movie | +0.0100 | +0.0069 | +0.0102 | +0.0070 | tie | tie |
| mental | mental | -0.0617 | -0.0046 | -0.0627 | -0.0046 | S | S |
| news | finance | -0.0027 | +0.0196 | -0.0112 | +0.0099 | S | S |
| sst2 | movie | -0.0009 | +0.0060 | -0.0010 | +0.0063 | S | S |

#### qwen-7b_newlayout / information（逐数据集 Δ）
| dataset | domain | ΔAcc(N) | ΔAcc(S) | ΔF1(N) | ΔF1(S) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0230 | -0.0153 | +0.0130 | -0.0068 | N | N |
| imdb | movie | -0.0002 | -0.0008 | +0.0003 | -0.0013 | tie | tie |
| imdb_sklearn | movie | +0.0008 | +0.0011 | +0.0036 | +0.0021 | tie | tie |
| mental | mental | -0.0691 | -0.0167 | -0.0748 | -0.0172 | S | S |
| news | finance | -0.0055 | +0.0082 | -0.0046 | +0.0076 | S | S |
| sst2 | movie | +0.0083 | -0.0125 | +0.0071 | -0.0088 | N | N |

### 3.2 execution（J/P）：逐数据集 Δ

#### llama-3b_newlayout / execution（逐数据集 Δ）
| dataset | domain | ΔAcc(J) | ΔAcc(P) | ΔF1(J) | ΔF1(P) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.1462 | +0.0754 | +0.0800 | +0.0471 | J | J |
| imdb | movie | -0.0037 | -0.0001 | -0.0043 | +0.0024 | tie | P |
| imdb_sklearn | movie | -0.0006 | +0.0036 | +0.0001 | +0.0063 | tie | P |
| mental | mental | +0.0247 | +0.1363 | +0.0461 | +0.2252 | P | P |
| news | finance | -0.0067 | -0.0370 | -0.0101 | -0.0381 | J | J |
| sst2 | movie | -0.0085 | -0.0274 | -0.0246 | +0.0028 | J | P |

#### qwen-3b_newlayout / execution（逐数据集 Δ）
| dataset | domain | ΔAcc(J) | ΔAcc(P) | ΔF1(J) | ΔF1(P) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0324 | -0.0273 | +0.0010 | -0.0115 | J | J |
| imdb | movie | +0.0004 | -0.0007 | +0.0004 | -0.0007 | tie | tie |
| imdb_sklearn | movie | +0.0101 | +0.0073 | +0.0102 | +0.0074 | tie | tie |
| mental | mental | -0.0318 | -0.0289 | -0.0319 | -0.0290 | tie | tie |
| news | finance | +0.0001 | +0.0168 | -0.0019 | -0.0014 | P | tie |
| sst2 | movie | +0.0120 | -0.0131 | +0.0127 | -0.0140 | J | J |

#### qwen-7b_newlayout / execution（逐数据集 Δ）
| dataset | domain | ΔAcc(J) | ΔAcc(P) | ΔF1(J) | ΔF1(P) | winner_acc | winner_f1 |
|---|---|---|---|---|---|---|---|
| fiqasa | finance | +0.0136 | -0.0188 | +0.0042 | -0.0112 | J | J |
| imdb | movie | -0.0001 | -0.0004 | -0.0004 | -0.0002 | tie | tie |
| imdb_sklearn | movie | +0.0019 | +0.0016 | +0.0033 | +0.0035 | tie | tie |
| mental | mental | -0.0472 | -0.0318 | -0.0505 | -0.0328 | P | P |
| news | finance | -0.0074 | +0.0101 | +0.0009 | +0.0026 | P | tie |
| sst2 | movie | +0.0031 | -0.0029 | -0.0000 | -0.0007 | J | tie |

领域层面（仅给方向性结论，不强行总结为单一规律）：
- information：llama/qwen-3b 在 finance 上更偏向 S 占优；但 qwen-7b 在 finance 上存在 N/S 分化不稳定（依赖具体数据集）。  
- execution：整体变化更小、tie 较多；更可能需要引入行为指标（格式遵循/拒答倾向/啰嗦程度等）来建立稳定联系。

建议：后续把“execution 的偏移”拆到更细粒度行为指标（拒答倾向/啰嗦程度/格式遵循等），再评估其与任务的对应关系。

---

## 4) 产物位置（用于追溯与画图）

- CSV：
  - `global_performance/performance_long.csv`
  - `global_performance/performance_pairwise.csv`
  - `global_performance/performance_summary.csv`
- 图：
  - `global_performance/plots/<model_root>/delta_accuracy_heatmap.png`
  - `global_performance/plots/<model_root>/delta_f1_heatmap.png`
  - `global_performance/plots/<model_root>/pushpull_*_<pair>.png`
  - `global_performance/plots/<model_root>/domain_bar_*_<pair>.png`
