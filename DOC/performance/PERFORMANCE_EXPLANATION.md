# Performance（性能）说明（newlayout，sentiment only）

> 本文用于说明 Stage-5 的“性能提升/下降”统计口径：在 **不改变评测数据集** 的前提下，回答性格微调后相对 base 的 **准确率（Accuracy）** 与 **宏平均 F1（Macro-F1）** 是提高还是下降。
>
> 与 `SHIFT_EXPLANATION.md` / `SIGNIFICANCE_EXPLANATION.md` / `EFFECT_EXPLANATION.md` 的关系：
> - shift：回答“预测分布往哪边偏”
> - significance/effect：回答“偏移是否可靠、偏移多大”
> - performance：回答“偏移是否带来任务性能提升（accuracy/F1）”

数据来源（newlayout）：
- 原始汇总：`*_newlayout/*/summaries/sentiment.csv`（Stage-2 输出）
- Stage-5 输出：`global_performance/performance_*.csv`（由 `stage5_global_performance.py` 生成）

---

## 1) 主指标（只看 sentiment）

我们只使用 `sentiment.csv` 中的 strict 指标：

- `accuracy_strict`
- `f1_macro_strict`

说明：strict 会对“输出无法解析为合法标签”的情况进行惩罚（等价于错判），因此更适合回答“性能是否真正提升”。

---

## 2) 相对 base 的提升/下降（Δ 指标）

对同一组（`model_root × pair × dataset × run`），定义：

- $\Delta Acc = Acc_{tuned} - Acc_{BASE}$
- $\Delta F1 = F1_{tuned} - F1_{BASE}$

其中：
- `BASE` 指 `sentiment.csv` 中 `model=BASE` 的行
- 两端性格模型是同一 `pair` 下的两个 `model`（例如 energy: E/I）

---

## 3) “有效提升/下降”阈值（bucket）

为了避免把极小波动当作结论，我们使用固定阈值：

- `threshold = 0.005`（0.5 个百分点）

对每个 Δ 指标分别分桶：
- `Δ >= +threshold` → `improve`
- `Δ <= -threshold` → `decline`
- 否则 → `tie`

> 备注：阈值只用于“胜率/计数”的解释；报告里仍保留原始 Δ 数值。

---

## 4) 领域（domain）划分（固定）

用于建立“性格 ↔ 任务领域”的联系：

- 电影域：`imdb, imdb_sklearn, sst2`
- 金融域：`fiqasa, news`
- 心理健康：`mental`

---

## 5) Stage-5 输出文件说明

### 5.1 `performance_long.csv`（最重要，明细）

行粒度：每个性格模型一行（不含 BASE）。

核心字段：
- 键：`model_root,pair,dataset,domain,run,model`
- BASE 指标：`base_accuracy_strict,base_f1_macro_strict`
- 模型指标：`accuracy_strict,f1_macro_strict`
- 差值：`delta_accuracy_strict,delta_f1_macro_strict`
- 分桶：`delta_acc_bucket,delta_f1_bucket`

### 5.2 `performance_pairwise.csv`（两端性格谁更好）

行粒度：每个 `model_root × pair × dataset` 一行。

核心字段：
- `model_a, model_b`：两端性格
- `adv_acc_a_minus_b` / `adv_f1_a_minus_b`：A 相对 B 的优势
- `winner_acc` / `winner_f1`：在阈值下的胜负（A/B/tie）

### 5.3 `performance_summary.csv`（领域级汇总）

行粒度：每个 `model_root × pair × model × domain` 一行。

核心字段：
- `n_datasets`：该领域包含的数据集数
- `mean_delta_acc / median_delta_acc`（同理 F1）
- `n_improve_* / n_decline_* / n_tie_*` 与 `improve_rate_*`

---

## 6) 图表解释（`global_performance/plots/`）

- Heatmap：快速看每个（pair/model）在各数据集上的 ΔAcc / ΔF1（红负蓝正）
- Push-Pull 四象限图：同一 pair 的两端性格在同一任务上的“推-拉”性能表现
  - (+,+)：两端都提升（更像“整体能力提升”）
  - (+,-)/(-,+)：一端提升一端下降（最像“性格-任务”映射信号）
  - (-,-)：两端都下降（可能是负迁移或输出格式问题）
- Domain bar：按领域聚合后的平均 Δ（附带 dataset 间标准差）

