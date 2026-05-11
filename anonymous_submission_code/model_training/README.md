# Model Training And Inference

This module trains personality-adapted models with DPO/LoRA and runs sentiment and benchmark inference.

Primary entrypoint:

```bash
python run_model_training.py --dimension information --model-path <base-model-checkpoint>
```

Supported modes:

- `--dimension energy|information|decision|execution` trains both subtypes for one MBTI dimension.
- `--pair TYPE_A TYPE_B` trains a custom personality pair, such as `ST NF` or `ENTP ISFJ`.

Generated checkpoints and results are written to runtime output directories and are not part of this submission package.
