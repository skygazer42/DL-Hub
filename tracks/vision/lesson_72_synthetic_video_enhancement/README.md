# Lesson 72: Synthetic Video Enhancement

This lesson restores clean grayscale clips after spatial blur and additive noise. The paired clean
sequence is generated alongside each degraded input, providing a deterministic video-enhancement
benchmark without downloaded media.

## Implementation

- `data.py` returns degraded clips with `clean` targets.
- `model.py` applies residual frame blocks and returns an `enhanced` sequence.
- `train.py` optimizes reconstruction loss and reports PSNR.

## Quick Run

```bash
python -m tracks.vision.lesson_72_synthetic_video_enhancement.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_72_synthetic_video_enhancement/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and a checkpoint, with finite reconstruction loss
and train/eval PSNR values.
