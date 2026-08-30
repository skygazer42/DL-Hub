# Lesson 69: Synthetic Video Restoration

This lesson restores a clean grayscale sequence from a synthetically degraded clip. The renderer
adds noise to a controlled moving scene, retaining the full clean sequence as supervision.

## Implementation

- `data.py` produces degraded/clean clip pairs with configurable frame count and noise level.
- `model.py` applies a residual restoration network and returns the restored video.
- `train.py` combines framewise L1 reconstruction with a temporal-consistency loss.

## Quick Run

```bash
python -m tracks.vision.lesson_69_synthetic_video_restoration.train \
  --epochs 1 --num-samples 48 --batch-size 4 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_69_synthetic_video_restoration/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and a checkpoint, with finite L1 and temporal
losses for both training and evaluation.
