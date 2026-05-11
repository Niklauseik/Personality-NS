# Result Processing

This module processes raw inference outputs, evaluates metrics, validates MBTI tendencies, and generates plots.

Primary entrypoints:

```bash
python process_results.py --results-root <result-root>
python process_results.py --results-root <model-root>
```

The processing pipeline supports legacy run roots with `pipeline_state.json` and new-layout model roots that contain `base/`.
