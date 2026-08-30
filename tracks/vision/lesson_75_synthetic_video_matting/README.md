# Lesson 75: Synthetic Video Matting

This lesson estimates a soft alpha matte for a moving foreground over a gradient background. The
renderer supplies the composited grayscale clip, a three-level trimap (`0`, `0.5`, `1`), and the
continuous alpha target for every frame.

## Implementation

- `data.py` generates video/trimap/alpha triples with a soft circular foreground.
- `model.py` concatenates frame and trimap channels and predicts per-frame alpha logits.
- `train.py` combines BCE and L1 alpha losses and reports matte MAE.

## Quick Run

```bash
python -m tracks.vision.lesson_75_synthetic_video_matting.train \
  --epochs 1 --num-samples 48 --batch-size 4 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_75_synthetic_video_matting/<run_name>/`. Success requires the
standard config, metrics, log, and checkpoint artifacts, finite BCE/L1 losses, and non-negative
train/eval alpha MAE.
