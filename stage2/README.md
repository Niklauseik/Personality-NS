# Stage-2

Stage-2 的实现代码放在本目录。

- CLI 入口（根目录）：`stage2_process_results.py`
- 实现：`stage2/process_results.py`
- Newlayout 实现：`stage2/newlayout.py`
- 兼容 shim（根目录）：`stage2_newlayout.py`
- 评估/作图：`stage2/evaluate_sentiment.py`、`stage2/evaluate_benchmarks.py`、`stage2/draw_charts.py`
- 清洗/纠正：`stage2/sentiment_get_invalid.py`、`stage2/sentiment_label_correct.py`、`stage2/sentiment_label_merge.py`、`stage2/sentiment_label_count.py`
- 显著性：`stage2/sentiment_significance.py`
