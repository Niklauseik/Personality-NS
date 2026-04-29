# Personality 项目说明

## 项目目标
基于 MBTI 四维度（E/I、S/N、T/F、J/P）进行 DPO 微调，产出“性格模型”，并与基座模型在情感分类与基准任务上对比评估，生成指标与可视化图表。

## 核心概念
- **维度与子类型映射**（来自 `common/pipeline_utils.py`）
  - energy: E / I
  - information: S / N
  - decision: T / F
  - execution: J / P
- **模型角色**
  - 基座模型（显示名：原始基座模型）
  - 训练后的性格模型（显示名：如 E性格模型、N性格模型 或 ENTP性格模型）

## 目录结构概览
```
datasets/
  training_raw/           # 原始 DPO 训练数据（json）
  dpo_converted/          # 旧版 CSV 格式 DPO 数据
  sentiment/              # 情感数据集
  benchmark/              # 基准评测数据集
stage1_train_and_test.py   # Stage-1 入口（CLI wrapper）
stage1_train_and_test_old_version.py # Stage-1（旧版本）入口（CLI wrapper）
stage2_process_results.py  # Stage-2 入口（CLI wrapper）
stage3_significance.py     # Stage-3 入口（CLI wrapper）
stage3_benchmark_drop_ttest.py # Stage-3 benchmark 能力下降单侧 t-test 入口
stage2_newlayout.py        # Stage-2（新目录结构）便捷入口
stage4_global_summaries.py # Stage-4（全局汇总）便捷入口
stage5_global_performance.py # Stage-5（全局性能汇总：sentiment）便捷入口
common/                    # 跨阶段共享代码
  pipeline_utils.py
stage1/                    # Stage-1 实现代码
  dpo_training_chat.py
  dpo_training_old_version.py
  run_sentiment.py
  run_benchmark.py
  train_and_test.py
  train_and_test_old_version.py
stage2/                    # Stage-2 实现代码
  process_results.py
  newlayout.py
  evaluate_sentiment.py
  evaluate_benchmarks.py
  draw_charts.py
  sentiment_significance.py
  sentiment_get_invalid.py
  sentiment_label_correct.py
  sentiment_label_merge.py
  sentiment_label_count.py
stage3/                    # Stage-3 实现代码
  significance.py
  benchmark_drop_ttest.py
stage4/                    # Stage-4 实现代码（全局汇总）
  global_summaries.py
stage5/                    # Stage-5 实现代码（全局性能汇总：sentiment）
  global_performance.py
COMMAND.md                 # 常用命令示例
global_summaries/          # 多模型聚合的全局汇总（显著性）
global_performance/        # 多模型聚合的全局汇总（性能）
```

## 数据说明
- **训练数据**（`datasets/training_raw/`）
  文件名：`en_<dimension>_<subtype>.json`，样本字段包括 `instruction` / `input` / `output`。
  示例：`en_energy_extraversion.json`、`en_information_intuition.json`。
- **情感数据**（`datasets/sentiment/`）
  模型推理使用 `text` 列；评估时标签列与数值映射如下（来自 `stage2/evaluate_sentiment.py`）：
  - imdb: `label`，`0 -> positive`，`1 -> negative`
  - imdb_sklearn: `label`，`0 -> negative`，`1 -> positive`
  - sst2: `label`，`0 -> negative`，`1 -> positive`
  - fiqasa: `answer`（字符串标签：`negative/positive/neutral`）
  - news: `label`，`0 -> bearish`，`1 -> bullish`，`2 -> neutral`
  - mental: `label`（字符串标签：`normal/depression`）
- **基准数据**（`datasets/benchmark/`）
  gsm8k / arc_easy / boolq。

## Stage-1：训练 + 推理
入口脚本：`stage1_train_and_test.py`
主要流程：
1. 训练性格模型（DPO + LoRA）
2. 运行 benchmark（可关）
3. 运行 sentiment（可跑多次）

常用参数：
- `--dimension energy/information/decision/execution`
- `--pair ENTP ISFJ` 或 `--pair ST NF`
- `--model-path <基座模型目录>`
- `--sentiment-runs N`
- `--no-benchmark` / `--no-base-sentiment`
- `--results-root <结果前缀>`

示例（节选自 `COMMAND.md`）：
```
python stage1_train_and_test.py --dimension information --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
python stage1_train_and_test.py --pair ST NF --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
python stage1_train_and_test.py --dimension decision --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2 --no-benchmark
```

## Stage-2：后处理 + 评估 + 画图
入口脚本：`stage2_process_results.py`
流程顺序：
1. 收集非法预测（`stage2/sentiment_get_invalid.py`）
2. 可选 GPT 纠正（`stage2/sentiment_label_correct.py`，依赖 OpenAI key）
3. 合并纠正结果（`stage2/sentiment_label_merge.py`）
4. 统计分布（`stage2/sentiment_label_count.py`）
5. 评估情感（`stage2/evaluate_sentiment.py`）
6. 评估 benchmark（`stage2/evaluate_benchmarks.py`）
7. 显著性检验（`stage2/sentiment_significance.py`）
8. 画图（`stage2/draw_charts.py`）

支持：
- 单个目录/多个目录：`--results-root`
- glob 自动发现：`--results-glob`
- 仅预览：`--dry-run`
- 出错继续：`--continue-on-error`

## 结果目录结构（旧结构/三层结构）
### 1) 单次 run 结构（legacy）
```
results-<model>-<pair>-run1/
  pipeline_state.json
  benchmark/<model_display>/*_results.csv
  sentiment/<dataset>/<model_display>/*_sentiment_results.csv
  sentiment/metrics_summary.csv
  sentiment/label_distribution_summary.txt
  plots/*.png
```

### 2) 旧目录三层结构（模型根目录/维度/run）
```
qwen-3b/
  decision/
    first_run/
      pipeline_state.json
    second_run/
      pipeline_state.json
```

`stage2_process_results.py` 会自动向下展开到 run 级目录。

## 结果目录结构（新结构 newlayout）
`stage2_newlayout.py` 通过 **model_root/base/** 自动识别新结构：

```
<model_root>/
  base/
    benchmark/...
  <dimension_or_pair>/
    <model_code>/
      sentiment/run-001/<dataset>/*_results.csv
      benchmark/*_results.csv
    summaries/
      sentiment.csv
      benchmark.csv
      sentiment_significance.csv
    plots/
      bar/*.png
      pie/*.png
```

## 评估产物
- 情感评估汇总：`sentiment/metrics_summary.csv`
- 情感分布统计：`sentiment/label_distribution_summary.txt`
- benchmark 汇总：`benchmark/benchmark_metrics_summary.csv`
- 显著性检验汇总（newlayout）：`<pair>/summaries/sentiment_significance.csv`
- 可视化：`plots/*.png` 或新结构下 `plots/bar`、`plots/pie`

## 训练实现要点
 - `stage1/dpo_training_chat.py`：直接读 raw json，构建 prompt/chosen/rejected，使用 TRL DPO + LoRA。
 - `stage1/dpo_training_old_version.py`：先生成 CSV（`datasets/dpo_converted/`），再训练（兼容旧流程）。
- 性格序列训练（如 ENTP）通过 `stage1/train_and_test.py` 中的 sequence 构建。

## 环境与依赖
推断依赖（本仓库未提供 requirements 文件）：
- torch, transformers, trl, peft, datasets
- pandas, numpy, scikit-learn
- scipy（显著性检验 p 值）
- matplotlib, tqdm
- openai（仅用于 `sentiment_label_correct.py`）

建议使用 GPU（训练/推理均为 LLM 任务）。
