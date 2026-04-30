# Cross-model Consistency Table

生成日期：2026-04-30

## 数据来源

本表直接从三个模型家族的真实结果文件计算：

- `llama-3b_newlayout/*/summaries/sentiment.csv`
- `qwen-3b_newlayout/*/summaries/sentiment.csv`
- `qwen-7b_newlayout/*/summaries/sentiment.csv`

只使用 `run=avg` 行。每个维度的 `BASE` 行使用同一个维度 summary 文件里的 base，不跨维度混用 base。

## 计算方法

### Shift

对每个数据集定义预测倾向轴 `s`：

- FiQA-SA / IMDb / IMDb-Sklearn / SST-2:
  `s = ratio_positive - ratio_negative`
- News:
  `s = ratio_bullish - ratio_bearish`
- Mental:
  `s = ratio_normal - ratio_depression`

注意：这里使用 summary 文件中的原始 `ratio_*` 列，不对去除 neutral 后的标签比例重新归一化。这样与真实预测分布一致，避免 neutral 变化改变倾向方向。

对每个模型家族、数据集、维度和两极性格，计算：

`Delta s = s_tuned - s_base`

然后记录两极的方向模式，例如 Energy E/I 维度记录为：

`sign(Delta s_E) / sign(Delta s_I)`

符号规则：

- `\(\checkmark\)`: 三个模型家族的方向模式完全一致
- `\(\triangle\)`: 两个模型家族的方向模式一致
- `\(\times\)`: 三个模型家族没有形成多数一致方向

### F1

F1 使用用户确认的口径：在同一数据集和同一 MBTI 维度下，比较两极性格模型的 `f1_macro_strict`，看哪一极更高。

例如 Energy E/I 维度：

- 若 `F1_E > F1_I`，该模型家族记为 `E`
- 若 `F1_I > F1_E`，该模型家族记为 `I`
- 若两者完全相等，记为 `tie`

本表不设置额外阈值；只按实际 Macro-F1 大小比较。跨模型一致性符号同 Shift。

## 结果表

| Dataset | Energy Shift | Energy F1 | Information Shift | Information F1 | Decision Shift | Decision F1 | Execution Shift | Execution F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FiQA-SA | \(\checkmark\) | \(\checkmark\) | \(\triangle\) | \(\triangle\) | \(\checkmark\) | \(\checkmark\) | \(\times\) | \(\checkmark\) |
| IMDb | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\triangle\) | \(\triangle\) |
| IMDb-Sklearn | \(\triangle\) | \(\checkmark\) | \(\triangle\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\triangle\) | \(\triangle\) |
| SST-2 | \(\checkmark\) | \(\checkmark\) | \(\triangle\) | \(\triangle\) | \(\checkmark\) | \(\checkmark\) | \(\triangle\) | \(\triangle\) |
| News | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\checkmark\) | \(\triangle\) |
| Mental | \(\triangle\) | \(\checkmark\) | \(\triangle\) | \(\checkmark\) | \(\triangle\) | \(\checkmark\) | \(\triangle\) | \(\checkmark\) |

## LaTeX 表格

```latex
\begin{table*}[htbp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Cross-model consistency of prediction tendency shifts and performance consequences. For each dataset and MBTI dimension, \(\checkmark\) denotes consistency across all three model families, \(\triangle\) denotes consistency in two model families, and \(\times\) denotes weak or inconsistent direction.}
\label{tab:combined_consistency}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{2}{c}{Energy E/I}
& \multicolumn{2}{c}{Information S/N}
& \multicolumn{2}{c}{Decision F/T}
& \multicolumn{2}{c}{Execution J/P} \\
\cmidrule(lr){2-3}
\cmidrule(lr){4-5}
\cmidrule(lr){6-7}
\cmidrule(lr){8-9}
Dataset
& Shift & F1
& Shift & F1
& Shift & F1
& Shift & F1 \\
\midrule
FiQA-SA      & \(\checkmark\) & \(\checkmark\) & \(\triangle\)  & \(\triangle\)  & \(\checkmark\) & \(\checkmark\) & \(\times\)     & \(\checkmark\) \\
IMDb         & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\triangle\)  & \(\triangle\) \\
IMDb-Sklearn & \(\triangle\)  & \(\checkmark\) & \(\triangle\)  & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\triangle\)  & \(\triangle\) \\
SST-2        & \(\checkmark\) & \(\checkmark\) & \(\triangle\)  & \(\triangle\)  & \(\checkmark\) & \(\checkmark\) & \(\triangle\)  & \(\triangle\) \\
News         & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\checkmark\) & \(\triangle\) \\
Mental       & \(\triangle\)  & \(\checkmark\) & \(\triangle\)  & \(\checkmark\) & \(\triangle\)  & \(\checkmark\) & \(\triangle\)  & \(\checkmark\) \\
\bottomrule
\end{tabular}
\end{table*}
```

## 复核明细

下表保留每个单元格背后的三模型结果。`Shift pattern` 的顺序与列名一致，例如 `Energy E/I` 下 `+/-` 表示 `E` 的 `Delta s` 为正、`I` 的 `Delta s` 为负。`F1 winner` 表示两极中 Macro-F1 更高的一极。

| Dataset | Dimension | Llama shift | Qwen-3B shift | Qwen-7B shift | Llama F1 winner | Qwen-3B F1 winner | Qwen-7B F1 winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FiQA-SA | Energy E/I | +/- | +/- | +/- | E | E | E |
| FiQA-SA | Information S/N | -/+ | +/+ | -/+ | S | S | N |
| FiQA-SA | Decision F/T | +/- | +/- | +/- | F | F | F |
| FiQA-SA | Execution J/P | +/+ | +/- | -/+ | J | J | J |
| IMDb | Energy E/I | +/- | +/- | +/- | E | E | E |
| IMDb | Information S/N | -/+ | -/+ | -/+ | N | N | N |
| IMDb | Decision F/T | +/- | +/- | +/- | F | F | F |
| IMDb | Execution J/P | -/+ | +/+ | -/+ | P | J | P |
| IMDb-Sklearn | Energy E/I | +/- | +/- | +/+ | E | E | E |
| IMDb-Sklearn | Information S/N | -/+ | +/+ | +/+ | N | N | N |
| IMDb-Sklearn | Decision F/T | +/- | +/- | +/- | F | F | F |
| IMDb-Sklearn | Execution J/P | -/+ | +/+ | +/+ | P | J | P |
| SST-2 | Energy E/I | +/- | +/- | +/- | E | E | E |
| SST-2 | Information S/N | -/+ | +/- | -/+ | N | S | N |
| SST-2 | Decision F/T | +/- | +/- | +/- | F | F | F |
| SST-2 | Execution J/P | -/+ | +/- | -/+ | P | J | J |
| News | Energy E/I | +/- | +/- | +/- | I | I | I |
| News | Information S/N | -/+ | -/+ | -/+ | S | S | S |
| News | Decision F/T | +/- | +/- | +/- | T | T | T |
| News | Execution J/P | -/- | -/- | -/- | J | P | P |
| Mental | Energy E/I | +/- | -/- | -/- | E | E | E |
| Mental | Information S/N | +/- | -/- | -/- | S | S | S |
| Mental | Decision F/T | -/- | -/+ | -/+ | T | T | T |
| Mental | Execution J/P | +/+ | -/- | -/- | P | P | P |
