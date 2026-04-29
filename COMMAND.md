# 训练与评估命令示例（Stage-1 / Stage-2 / Stage-3）

从头到尾操作说明（云端 Stage-1 + 本地 Stage-2 + 本地 Stage-3）。

## 0) 本地准备
- 在仓库根目录操作（包含 `stage1_train_and_test.py` / `stage2_process_results.py` / `stage3_significance.py`）。
- 本地 Stage-2/Stage-3 只需要“结果目录”，不需要本地跑 Stage-1。

## 1) 上传数据到云端并解压
```bash
# 上传 datasets.zip 到云端工作目录
scp -P 48853 ./datasets.zip root@region-46.seetacloud.com:/root/autodl-tmp/personality/

# 登录云端
ssh -p 48853 root@region-46.seetacloud.com

# 解压数据（示例解压到 datasets/）
cd /root/autodl-tmp/personality
unzip -o datasets.zip -d datasets
```

## 2) 上传 Stage-1 相关代码到云端
```bash
# 简单起见：把仓库的入口脚本 + 实现目录都传上去
# - 入口：stage*_*.py
# - 实现：common/ stage1/ stage2/ stage3/
scp -P 48853 stage*_*.py root@region-46.seetacloud.com:/root/autodl-tmp/personality/
scp -P 48853 -r common stage1 stage2 stage3 root@region-46.seetacloud.com:/root/autodl-tmp/personality/
```

## 3) 云端运行 Stage-1（训练 + sentiment + benchmark）
```bash
# 示例：Information 维度（S/N）
python stage1_train_and_test.py --dimension information --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2

# 示例：指定两个字母（ST + NF）
python stage1_train_and_test.py --pair ST NF --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2

# 可选：关闭 benchmark（只跑 sentiment）
python stage1_train_and_test.py --dimension information/energy/decision/execution --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2 --no-benchmark

# 可选：不跑基座模型的 sentiment（只跑性格模型的 sentiment）
python stage1_train_and_test.py --dimension information/energy/decision/execution --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2 --no-base-sentiment
```

## 4) 下载云端结果到本地仓库根目录
```bash
# 结果目录结构约定：三层固定
#   <模型根目录>/<维度>/<run目录>/pipeline_state.json
# 例如：
#   qwen-3b/information/first_run/pipeline_state.json
#   qwen-3b/decision/second_run/pipeline_state.json

# 把某个模型的全部结果下载到本地“仓库根目录”（不要下载到 ./results 下）
scp -P 48853 -r root@region-46.seetacloud.com:/root/autodl-tmp/personality/results/qwen-3b .

# 如果 benchmark 在云端单独输出（例如 results/benchmark），也下载到仓库根目录，然后你可以手动放到对应 run 目录里
scp -P 48853 -r root@region-46.seetacloud.com:/root/autodl-tmp/personality/results/benchmark .
```

## 5) 本地运行 Stage-2（后处理 + 评估 + 画图）
```bash
# Stage-2 支持传不同层级：
# - 传模型根目录：处理该模型下全部维度 + 全部 run
python stage2_process_results.py --results-root qwen-3b

# - 传维度目录：只处理该维度下全部 run
python stage2_process_results.py --results-root qwen-3b/information

# - 传 run 目录：只处理这一次 run（目录内需有 pipeline_state.json）
python stage2_process_results.py --results-root qwen-3b/information/first_run

# 兼容旧结构：如果你直接传的是一个包含 pipeline_state.json 的目录（如 results-NS-first-run），也会直接处理
python stage2_process_results.py --results-root results-NS-first-run

# newlayout 会额外输出 sentiment 显著性检验汇总（Stuart-Maxwell / McNemar）
# 位置：<pair>/summaries/sentiment_significance.csv
```

## 6) 本地运行 Stage-3（显著性汇总 + 热力图 + 小提琴图）
```bash
# 处理全部 newlayout 模型目录
python stage3_significance.py --model-glob "*_newlayout"

# 带参数：调整 p 阈值 / p 下限，并仅预览要处理的目录
python stage3_significance.py --model-glob "*_newlayout" --p-threshold 0.01 --min-p 1e-300 --dry-run

# 只处理某一个模型目录
python stage3_significance.py --model-root qwen-3b_newlayout

# 带参数：只处理指定模型，并调整阈值
python stage3_significance.py --model-root qwen-3b_newlayout --p-threshold 0.01 --min-p 1e-200

# Benchmark 能力下降单侧 t-test（Delta Acc = tuned - base，H1: mean(delta) < 0）
# 默认读取 BENCHMARK_CAPABILITY_CHANGES_BY_DATASET.md，输出到 global_summaries/
python stage3_benchmark_drop_ttest.py

# 可选：在 Markdown 表中也显示 Overall p-drop
python stage3_benchmark_drop_ttest.py --include-overall-p-in-markdown
```

## 7) 本地运行 Stage-4（全局汇总：跨所有 *_newlayout）
```bash
# 生成全局 CSV（默认输出到 global_summaries/）
python stage4_global_summaries.py --root . --output-dir global_summaries

# 带参数：只画图（不重写 CSV），并指定效应量指标
python stage4_global_summaries.py --root . --output-dir global_summaries --no-summarize --plot --effect-metric effect_js

# 带参数：生成 CSV + 同时画图
python stage4_global_summaries.py --root . --output-dir global_summaries --plot --effect-metric effect_cramers_v
```

## 8) 本地运行 Stage-5（性能提升/下降：sentiment only，accuracy + macro-F1）
```bash
# 扫描仓库下所有 *_newlayout/*/summaries/sentiment.csv，输出到 global_performance/
# 默认只用 run=avg；“有效提升/下降”阈值默认 0.005 = 0.5pp

# 只生成 CSV
python stage5_global_performance.py --root . --output-dir global_performance

# 生成 CSV + 图表（heatmap / 四象限 push-pull / 领域汇总图）
python stage5_global_performance.py --root . --output-dir global_performance --plot

# 可选：修改阈值（例如 1pp）
python stage5_global_performance.py --root . --output-dir global_performance --threshold 0.01 --plot
```
