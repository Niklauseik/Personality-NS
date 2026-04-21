# Performance Synthesis

> 本文是对 [PERFORMANCE_REPORT.md](C:/Users/BiaoPuYun/Personality-NS/DOC/performance/PERFORMANCE_REPORT.md) 的高层总结，只看 `sentiment` 任务、`run=avg`、`strict` 指标（`Accuracy` / `Macro-F1`）。
>
> 核心目标不是重复逐数据集结果，而是回答四个更高层的问题：
> 1. 哪个 MBTI 维度在性能层面的“推拉”最强？
> 2. 两个对立性格是否会带来相反方向的能力变化？
> 3. 这种模式在不同模型家族上是否稳定？
> 4. 各维度是否形成了“一个性格更擅长某领域、另一个性格更擅长另一领域”的结构化映射？

---

## 1) 总结先行

### 1.1 最强结论：`decision (F/T)` 是性能层面最清晰、最稳定、最可解释的维度

- 在四个 MBTI 维度里，`decision` 的两极性能差距最大。
- 它也是“两个对立性格朝相反方向变化”最常见的维度。
- 更关键的是，它形成了最清晰的**领域映射**：
  - `movie`：`F` 明显优于 `T`
  - `mental`：`T` 明显优于 `F`
- 这种模式在三个模型家族上都能看到，只是强度不同。

这意味着 `F/T` 不是“随机地一边强一边弱”，而更像是**稳定的领域适配分化**。

### 1.2 次强结论：`energy (E/I)` 在电影域上稳定，但更像“E 整体更强”，而不是强烈的双向领域互补

- `E/I` 的性能推拉强度仅次于 `F/T`。
- 在 `movie` 上，`E` 稳定优于 `I`。
- 在 `mental` 上，`E` 也通常优于 `I`，因此它不像 `F/T` 那样形成“一个领域归一端、另一个领域归另一端”的鲜明交换结构。

### 1.3 `information (N/S)` 与 `execution (J/P)` 都有结构，但稳定性明显更弱

- `information` 的模式是：
  - `S` 更偏向 `finance + mental`
  - `N` 只在 `movie` 上表现出较弱优势，而且 tie 较多
- `execution` 的模式是：
  - `J` 更偏向 `finance`
  - `P` 更偏向 `mental`
  - `movie` 上大量 tie，说明这个维度在标签指标下不够敏感

### 1.4 模型角度：三个模型家族里，最强维度都是 `decision`

- `Llama-3.2-3B`：`decision` 最强，`energy` 紧随其后，整体变化幅度最大
- `Qwen2.5-3B`：`decision` 最清晰，且最接近“标准推拉结构”
- `Qwen2.5-7B`：`decision` 仍然最强，但总体幅度比 `Qwen2.5-3B` 更小，说明更大模型更稳

---

## 2) 哪个维度最明显？

这里用三个指标衡量维度强弱：

- `mean |Δ_left - Δ_right|`：两端性格的平均性能差距，越大说明“推拉”越强
- `raw opposite-sign`：两端相对 base 的性能变化是否异号，越多说明越常出现“相反方向变化”
- `strict push-pull`：在 `0.005` 阈值下，一端 `improve`、另一端 `decline` 的次数，越多说明越不是小波动

### 2.1 维度级总体对比

| Dimension | Mean abs gap Acc | Mean abs gap F1 | Raw opposite-sign Acc | Raw opposite-sign F1 | Strict push-pull Acc | Strict push-pull F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision | 0.0633 | 0.0527 | 17 / 18 | 16 / 18 | 12 / 18 | 11 / 18 |
| Energy | 0.0501 | 0.0390 | 13 / 18 | 14 / 18 | 8 / 18 | 7 / 18 |
| Information | 0.0283 | 0.0286 | 9 / 18 | 11 / 18 | 5 / 18 | 4 / 18 |
| Execution | 0.0233 | 0.0202 | 7 / 18 | 6 / 18 | 4 / 18 | 1 / 18 |

结论很直接：

- `decision` 在三个指标上都是第一。
- `energy` 明显第二。
- `information` 和 `execution` 都弱得多，其中 `execution` 在 `F1` 上几乎很少形成严格 push-pull。

如果论文要回答“哪个维度最明显”，最稳的写法就是：

> Across model families and datasets, `decision (F/T)` shows the strongest performance-level push-pull pattern, with the largest average pole-to-pole gap and the highest coverage of opposite-direction changes.

---

## 3) 对立性格是否会带来相反方向的能力变化？

答案是：**会，但强度高度依赖维度。**

### 3.1 `decision (F/T)`：最接近“相反性格 -> 相反能力变化”

- `Accuracy` 上，`17/18` 个 `(model family × dataset)` 单元出现异号变化
- `Macro-F1` 上，`16/18` 个单元出现异号变化
- 进一步用阈值约束后，仍有 `12/18` 个 `Acc` 单元和 `11/18` 个 `F1` 单元形成严格的一升一降

这说明：

- `F/T` 的差异不是“谁都一起涨或一起跌”
- 而是更常见地表现为**一端被推高、另一端被拉低**

### 3.2 `energy (E/I)`：也常相反，但更像单侧偏强

- `Acc` 异号覆盖 `13/18`
- `F1` 异号覆盖 `14/18`

这个维度也确实经常相反，但它的领域结构不像 `F/T` 那么“交换式”。很多时候更像是：

- `E` 在关键领域上更经常赢
- `I` 则在部分金融数据集上反超

### 3.3 `information (N/S)` 与 `execution (J/P)`：相反变化并不稳定

- `information`：异号只达到 `9/18`（Acc）和 `11/18`（F1）
- `execution`：更低，仅 `7/18`（Acc）和 `6/18`（F1）

因此从“相反能力变化”这个角度，`decision` 最强，`energy` 次之，另外两个维度不适合作为主结论的核心支撑。

---

## 4) 模型家族角度：哪个家族最明显？

### 4.1 每个模型家族里，最强维度都是 `decision`

| Model family | Strongest dimension | Mean abs gap Acc | Mean abs gap F1 | Second strongest |
| --- | --- | ---: | ---: | --- |
| Llama-3.2-3B | Decision | 0.0988 | 0.0876 | Energy |
| Qwen2.5-3B | Decision | 0.0507 | 0.0417 | Energy |
| Qwen2.5-7B | Decision | 0.0402 | 0.0290 | Information / Energy close behind, but still lower |

这给出两个很有价值的高层结论：

1. `decision` 不是只在某一个模型家族上偶然成立，它在三个家族里都是最强维度。
2. 模型越大，模式通常仍保留，但幅度更小。

### 4.2 `Llama-3.2-3B` 的对比幅度最大

从平均两极差距看，`Llama-3.2-3B` 的极化最强：

- `decision`: `0.0988 Acc / 0.0876 F1`
- `energy`: `0.0937 Acc / 0.0736 F1`

这意味着 `Llama-3.2-3B` 更容易被性格信号推离 base 行为；它的结果更“戏剧化”，也更容易出现大幅正负变化。

### 4.3 `Qwen2.5-3B` 的 `decision` 最接近“干净推拉”

在 `Qwen2.5-3B` 上：

- `decision` 的 `raw opposite-sign` 达到 `6/6`（Acc）和 `6/6`（F1）
- 严格 push-pull 也有 `5/6`（Acc）和 `4/6`（F1）

这说明它不是单纯由极端个例拉出来的，而是几乎在所有数据集上都按同样逻辑工作。

### 4.4 `Qwen2.5-7B` 保留模式，但更保守

`Qwen2.5-7B` 仍然把 `decision` 作为最强维度，但 gap 比 `Qwen2.5-3B` 更小：

- `decision`：`0.0402 / 0.0290`
- `energy`：`0.0205 / 0.0150`
- `execution`：`0.0120 / 0.0060`

这与前面 shift/effect 部分“更大模型更稳”的结论是相互一致的。

---

## 5) 领域角度：是否出现“一个性格更适合一个领域，另一个性格更适合另一个领域”？

答案是：**有，但强度差异很大。最清楚的是 `decision`，其次是 `execution`，再其次是 `information`。`energy` 更像单侧优势，不像对称互补。**

### 5.1 `decision (F/T)`：最强、最漂亮的领域交换结构

#### movie：`F` 稳定强于 `T`

- `winner_acc`: `F 8 / 9`，`tie 1 / 9`
- `winner_f1`: `F 9 / 9`
- 跨家族平均领域增益：
  - `F`: `+0.0161 Acc`, `+0.0158 F1`
  - `T`: `-0.0297 Acc`, `-0.0282 F1`

#### mental：`T` 稳定强于 `F`

- `winner_acc`: `T 3 / 3`
- `winner_f1`: `T 3 / 3`
- 跨家族平均领域增益：
  - `F`: `-0.0322 Acc`, `-0.0414 F1`
  - `T`: `+0.0124 Acc`, `+0.0114 F1`

#### finance：不能粗暴说“只偏 F”或“只偏 T”

- `winner_acc`: `F 3 / 6`, `T 3 / 6`
- `winner_f1`: `F 3 / 6`, `T 3 / 6`
- 但平均差仍偏向 `F`：
  - `F`: `+0.0139 Acc`, `+0.0023 F1`
  - `T`: `-0.0158 Acc`, `-0.0123 F1`

这说明 finance 内部并不单一：

- `FiQA-SA` 更偏 `F`
- `News` 更偏 `T`

因此，`decision` 的最佳高层表述是：

> `F/T` forms a clean movie-vs-mental specialization, while finance further splits into sub-domains rather than supporting one universal pole.

### 5.2 `energy (E/I)`：电影域稳定偏 `E`，但不是对称交换

#### movie：`E` 明显占优

- `winner_acc`: `E 7 / 9`, `tie 2 / 9`
- `winner_f1`: `E 7 / 9`, `tie 2 / 9`
- 跨家族平均领域增益：
  - `E`: `+0.0151 Acc`, `+0.0140 F1`
  - `I`: `-0.0125 Acc`, `-0.0121 F1`

#### mental：也是 `E` 占优

- `winner_acc`: `E 3 / 3`
- `winner_f1`: `E 3 / 3`
- 平均上两端都偏负，但 `E` 明显“更不差”：
  - `E`: `-0.0094 Acc`, `-0.0038 F1`
  - `I`: `-0.0382 Acc`, `-0.0453 F1`

#### finance：明显不稳定

- `winner_acc`: `E 3 / 6`, `I 3 / 6`
- `winner_f1`: `E 3 / 6`, `I 2 / 6`, `tie 1 / 6`

所以 `energy` 的结论不是“E 擅长 A 域、I 擅长 B 域”，而是：

> `E/I` shows a robust movie-domain separation, but outside movie it behaves more like a relative strength asymmetry than a clean cross-domain specialization.

### 5.3 `information (N/S)`：`S` 偏 finance + mental，`N` 在 movie 只有弱优势

#### finance：`S` 明显占优

- `winner_acc`: `S 5 / 6`
- `winner_f1`: `S 5 / 6`
- 平均领域增益：
  - `N`: `-0.0099 Acc`, `-0.0102 F1`
  - `S`: `+0.0209 Acc`, `+0.0107 F1`

#### mental：`S` 完全占优

- `winner_acc`: `S 3 / 3`
- `winner_f1`: `S 3 / 3`
- 平均领域增益：
  - `N`: `-0.0549 Acc`, `-0.0656 F1`
  - `S`: `+0.0013 Acc`, `+0.0027 F1`

#### movie：`N` 只有弱优势，且 tie 很多

- `winner_acc`: `N 4 / 9`, `S 1 / 9`, `tie 4 / 9`
- `winner_f1`: `N 5 / 9`, `S 1 / 9`, `tie 3 / 9`
- 平均领域增益：
  - `N`: `-0.0024 Acc`, `+0.0025 F1`
  - `S`: `-0.0096 Acc`, `-0.0120 F1`

因此 `information` 可以写成：

> `N/S` does show a domain split, but it is asymmetric: `S` is clearly better on finance and mental-health tasks, whereas `N` only has a weak and tie-heavy edge on movie tasks.

### 5.4 `execution (J/P)`：`J` 偏 finance，`P` 偏 mental，但整体灵敏度最低

#### finance：`J` 更强

- `winner_acc`: `J 4 / 6`, `P 2 / 6`
- `winner_f1`: `J 4 / 6`, `tie 2 / 6`
- 平均领域增益：
  - `J`: `+0.0297 Acc`, `+0.0124 F1`
  - `P`: `+0.0032 Acc`, `-0.0021 F1`

#### mental：`P` 更强

- `winner_acc`: `P 2 / 3`, `tie 1 / 3`
- `winner_f1`: `P 2 / 3`, `tie 1 / 3`
- 平均领域增益：
  - `J`: `-0.0181 Acc`, `-0.0121 F1`
  - `P`: `+0.0252 Acc`, `+0.0545 F1`

#### movie：大量 tie

- `winner_acc`: `tie 6 / 9`, `J 3 / 9`
- `winner_f1`: `tie 5 / 9`, `P 3 / 9`, `J 1 / 9`
- 平均领域增益接近 0：
  - `J`: `+0.0016 Acc`, `-0.0003 F1`
  - `P`: `-0.0036 Acc`, `+0.0008 F1`

所以 `execution` 虽然不是最强维度，但它其实有一个可以写进论文的中等强度结构：

> `J/P` forms a finance-vs-mental split, but its effect on standard label metrics is much weaker than `F/T`, especially on movie tasks where ties dominate.

---

## 6) 可以直接写进论文的高层结论

下面这些表述是当前结果最有支撑力的版本。

### 6.1 关于“哪个维度最明显”

> Among the four MBTI dimensions, `decision (F/T)` exhibits the strongest performance-level contrast. It has the largest average pole-to-pole gap (`0.0633` in Accuracy and `0.0527` in Macro-F1) and the highest coverage of opposite-direction changes (`17/18` and `16/18` raw opposite-sign cases, respectively).

### 6.2 关于“对立性格是否带来相反的能力变化”

> Yes, but this pattern is dimension-dependent. It is strongest for `decision`, secondary for `energy`, and substantially weaker for `information` and `execution`.

### 6.3 关于“是否形成领域互补”

> The clearest domain specialization appears in `decision`: `F` consistently favors movie tasks, whereas `T` consistently favors mental-health tasks. `execution` shows a weaker but interpretable finance-vs-mental split (`J` better on finance, `P` better on mental), while `energy` and `information` are less symmetric and behave more like one-sided or partial specializations.

### 6.4 关于“模型家族差异”

> The same qualitative hierarchy holds across all three model families: `decision` is always the strongest performance-sensitive dimension. However, the magnitude of the effect shrinks with model scale, with `Llama-3.2-3B` showing the strongest polarization and `Qwen2.5-7B` showing the most conservative version of the same pattern.

---

## 7) 结论边界

这部分结论只针对：

- `sentiment` 任务
- `run=avg`
- `strict` 指标
- 当前 6 个数据集的领域划分

因此最稳的说法是：

- 我们已经看到**稳定的性能层面结构**
- 其中 `decision` 最强，`energy` 次之
- `information` 与 `execution` 更依赖具体领域与数据集

但如果要把 `execution` 进一步做强，后续最好补充：

- 拒答倾向
- 格式遵循
- 冗长度 / 啰嗦程度
- 推理链组织度

这类行为指标，因为 `J/P` 可能更像“输出风格组织方式”的差异，而不是传统标签准确率的强信号。

---

## 8) 数据来源

- [PERFORMANCE_REPORT.md](C:/Users/BiaoPuYun/Personality-NS/DOC/performance/PERFORMANCE_REPORT.md)
- [performance_long.csv](C:/Users/BiaoPuYun/Personality-NS/global_performance/performance_long.csv)
- [performance_pairwise.csv](C:/Users/BiaoPuYun/Personality-NS/global_performance/performance_pairwise.csv)
- [performance_summary.csv](C:/Users/BiaoPuYun/Personality-NS/global_performance/performance_summary.csv)
