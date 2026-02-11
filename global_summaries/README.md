# 全局汇总（global_summaries）

这个目录用于存放**跨多个模型根目录（`*_newlayout/`）**的“全局聚合”结果，避免与每个模型/维度目录下的 `summaries/`（Stage-2 输出）以及每个模型的 `stage3/summaries/`（Stage-3 输出）混淆。

生成方式：

- 生成 CSV：`python stage4_global_summaries.py`
- 画图（可选）：`python stage4_global_summaries.py --plot`

## `significance_long.csv` 是什么？

行级表：把仓库里所有 `*_newlayout/*/summaries/sentiment_significance.csv` 直接拼接在一起（每行对应某个 `pair`/`model`/`run` 在某个 `dataset` 上的一次显著性检验结果）。

常用字段（以实际列为准）：

- `model_root`：模型根目录名，例如 `qwen-7b_newlayout`
- `pair`：维度/配对名称（例如 `energy`、`information`、`ST-NF` 等）
- `model`：模型代号（通常来自 `pair_root/<model_code>/...` 的子目录名，如 `F/T`、`ST/NF` 等）
- `run`：运行号（如 `run-001`、`run-002`）
- `dataset`：数据集标识（如 `imdb`、`sst2` 等）
- `test`：使用的检验方法名（如 `stuart_maxwell`、`mcnemar_exact`）
- `p_value`：p 值
- `effect_cramers_v` / `effect_tv` / `effect_js`：效应量/分布差异指标
- `n_total` / `n_used` / `n_dropped`：样本数、有效样本数、丢弃数
- `source_file`：该行来源的原始 CSV 路径（用于追溯）

## `significance_summary.csv` 是什么？

聚合表：对 `significance_long.csv` 按 `(model_root, pair, model, dataset, test)` 分组进行汇总，便于快速比较不同模型/维度/数据集的整体差异水平。

主要字段：

- `model_root` / `pair` / `model` / `dataset` / `test`：分组键
- `n_rows`：该分组包含的行数（通常对应不同 `run` 的次数）
- `n_sig_p_lt_0.05`：p 值 < 0.05 的条目数
- `sig_rate`：显著率 = `n_sig_p_lt_0.05 / n_rows`
- `p_min` / `p_median` / `p_max`：p 值统计量
- `effect_*_median`：效应量中位数（更稳健，减少极端值影响）

## 图表输出（可选）

当使用 `--plot` 时，会按 `model_root` 分别生成图表，避免不同模型根目录的数据混在一起：

- `global_summaries/plots/<model_root>/significance_heatmap.png`
- `global_summaries/plots/<model_root>/effect_<metric>.png`
