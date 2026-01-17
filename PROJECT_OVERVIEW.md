# Personality 项目说明

## 项目目标
基于 MBTI 四维度（E/I、S/N、T/F、J/P）进行 DPO 微调，产出“性格模型”，并与基座模型在情感分类与基准任务上对比评估，生成指标与可视化图表。

## 核心概念
- **维度与子类型映射**（来自 `pipeline_utils.py`）
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
dpo_training_chat.py       # 新版 DPO 训练（直接读 raw json）
dpo_training_old_version.py# 旧版 DPO 训练（读 dpo_converted csv）
stage1_train_and_test.py   # Stage-1：训练+推理
stage1_train_and_test_old_version.py
stage2_process_results.py  # Stage-2：后处理+评估+作图（旧结构/混合结构）
stage2_newlayout.py        # Stage-2（新目录结构）
run_sentiment.py           # 仅跑情感推理
run_benchmark.py           # 仅跑 benchmark 推理
evaluate_sentiment.py      # 情感评估汇总
evaluate_benchmarks.py     # benchmark 评估汇总
draw_charts.py             # 画情感分布图
sentiment_significance.py  # 情感显著性检验（Stuart-Maxwell / McNemar）
sentiment_*                # 情感结果清洗/纠正/合并/统计
command.txt                # 常用命令示例
```

## 数据说明
- **训练数据**（`datasets/training_raw/`）
  文件名：`en_<dimension>_<subtype>.json`，样本字段包括 `instruction` / `input` / `output`。
  示例：`en_energy_extraversion.json`、`en_information_intuition.json`。
- **情感数据**（`datasets/sentiment/`）
  模型推理使用 `text` 列；评估时标签列与数值映射如下（来自 `evaluate_sentiment.py`）：
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

示例（节选自 `command.txt`）：
```
python stage1_train_and_test.py --dimension information --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
python stage1_train_and_test.py --pair ST NF --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
python stage1_train_and_test.py --dimension decision --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2 --no-benchmark
```

## Stage-2：后处理 + 评估 + 画图
入口脚本：`stage2_process_results.py`
流程顺序：
1. 收集非法预测（`sentiment_get_invalid.py`）
2. 可选 GPT 纠正（`sentiment_label_correct.py`，依赖 OpenAI key）
3. 合并纠正结果（`sentiment_label_merge.py`）
4. 统计分布（`sentiment_label_count.py`）
5. 评估情感（`evaluate_sentiment.py`）
6. 评估 benchmark（`evaluate_benchmarks.py`）
7. 显著性检验（`sentiment_significance.py`）
8. 画图（`draw_charts.py`）

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

### 2) 三层结构（模型根目录/维度/run）
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
- `dpo_training_chat.py`：直接读 raw json，构建 prompt/chosen/rejected，使用 TRL DPO + LoRA。
- `dpo_training_old_version.py`：先生成 CSV（`datasets/dpo_converted/`），再训练（兼容旧流程）。
- 性格序列训练（如 ENTP）通过 `stage1_train_and_test.py` 中的 sequence 构建。

## 环境与依赖
推断依赖（本仓库未提供 requirements 文件）：
- torch, transformers, trl, peft, datasets
- pandas, numpy, scikit-learn
- scipy（显著性检验 p 值）
- matplotlib, tqdm
- openai（仅用于 `sentiment_label_correct.py`）

建议使用 GPU（训练/推理均为 LLM 任务）。
