# Lesson 12: Compact Layout-to-Image Generation

This lesson maps a semantic layout tensor to a grayscale image. Each offline synthetic sample
contains one to three class-specific rectangular regions; class intensity determines the rendered
target, with a small amount of pixel noise added for reconstruction practice.

## Implementation

- `data.py` produces multi-channel class layouts, paired targets, and a seeded train/validation split.
- `model.py` uses a compact convolutional encoder-decoder-style stack to predict one output channel.
- `train.py` optimizes binary cross-entropy and reports `train_bce` and `val_bce` each epoch.

## Quick Run

```bash
python -m tracks.generative.lesson_12_compact_layout_to_image.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/generative/lesson_12_compact_layout_to_image/<run_name>/`. A
successful smoke run writes `config.json`, `metrics.jsonl`, `samples.pt`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. `samples.pt` contains `layout`, `target`, and `prediction` tensors, and
the metrics file must contain finite training and validation BCE values.
