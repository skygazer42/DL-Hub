# Lesson 60: Synthetic Image Deraining

This lesson restores clean RGB shape scenes after adding a separately rendered layer of diagonal
rain streaks. Rain count, length, and strength are configurable, and the paired clean/rainy data is
generated offline.

## Implementation

- `data.py` returns a rainy image with clean-image and rain-layer targets.
- `model.py` uses a residual CNN with restored-image and rain-layer outputs.
- `train.py` supervises both outputs and reports clean reconstruction PSNR.

## Quick Run

```bash
python -m tracks.vision.lesson_60_synthetic_image_deraining.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_60_synthetic_image_deraining/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`, with finite
reconstruction/rain losses and train/eval PSNR values.
