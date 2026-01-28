# Stage-3 绘图说明

本文件说明 stage3 生成的图与汇总表含义，方便快速理解图像内容。

## 适用范围
- 仅处理 new-layout 结构（包含 `base/` 的模型根目录）
- 每个模型的输出都在其根目录下的 `stage3/` 里

## 已生成输出位置
- `llama-3b_newlayout/stage3/`
- `qwen-3b_newlayout/stage3/`
- `qwen-7b_newlayout/stage3/`

## 数据来源
- 每个维度目录下的 `summaries/sentiment_significance.csv` (Stage-2 输出)

## 汇总表（每个模型的 stage3/summaries/ 内）
- `significance_long.csv`
  - 行级数据：维度 × 数据集 × run × 模型
- `significance_summary.csv`
  - 对同一维度/数据集/模型 **跨 run 取平均**
  - 包含 `p_value`、`effect_*` 与 `conclusion`
- `significance_dimension_summary.csv`
  - 进一步对维度/数据集 **跨模型与 run 取平均**
  - 这是热力图的直接数据来源

## 图 1：p-value 热力图（每个模型的 stage3/plots/heatmap/ 内）
- 文件：`pvalue_heatmap.png`
- 横轴：数据集
- 纵轴：性格维度
- 颜色：`-log10(p)`，p-value 使用下限 1e-300，避免出现 0
- 参考：`p < 0.05` 对应 `-log10(p) > 1.30`

## 图 2：Effect size 小提琴图（每个模型的 stage3/plots/effect/ 内）
- 每个 effect 指标单独出图：
  - `effect_cramers_v`
  - `effect_tv`
  - `effect_js`
- 横轴：数据集
- 颜色：性格维度
- 小提琴展示 **run 与模型的分布**，便于观察稳定性

## 备注
- 若某数据集或维度缺数据，会在热力图中显示为浅灰色
- 可结合 `SIGNIFICANCE_EXPLANATION.md` 与 `EFFECT_EXPLANATION.md` 阅读统计含义
