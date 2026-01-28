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
# 简单起见：把仓库里用到的脚本都传上去（也可以只传 stage1/run_* / pipeline_utils.py 等）
scp -P 48853 ./*.py root@region-46.seetacloud.com:/root/autodl-tmp/personality/
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

# 只处理某一个模型目录
python stage3_significance.py --model-root qwen-3b_newlayout
```

## Stage-1：启动训练（Information 维度：S/N）
```bash
# 使用本地 Qwen 模型目录（例：qwen2.5-7B-Instruct）
# - benchmark：默认开启（跑 1 次）
# - sentiment：跑 2 次
python stage1_train_and_test.py --dimension information --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
```

## Stage-1：启动训练（不指定维度，指定类型/组合）
```bash
# 训练一对自定义类型（例：ENTP vs ISFJ）
python stage1_train_and_test.py --pair ENTP ISFJ --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2

# 也支持用字母组合（例：ST vs NF）
python stage1_train_and_test.py --pair ST NF --model-path ./qwen2.5-7B-Instruct --sentiment-runs 2
```

## Stage-2：处理结果 + 评估 + 画图
```bash
# 处理单个 results 目录
python stage2_process_results.py --results-root results-qwen2.5-7B-Instruct-N-S-run1

# 处理多个 results 目录（示例：一次性处理两组结果）
python stage2_process_results.py --results-root results-qwen2.5-7B-Instruct-N-S-run1 results-qwen2.5-7B-Instruct-N-S-run2

# 用 glob 自动发现多个 results 目录（可重复指定多个 pattern）
python stage2_process_results.py --results-glob "results-*-run*" --continue-on-error

# 仅查看将会处理哪些目录（不真正执行）
python stage2_process_results.py --results-glob "results-*" --dry-run
```
