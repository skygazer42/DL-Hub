# Lesson 68: Synthetic Video Frame Interpolation

This lesson predicts the middle RGB frame between two endpoint frames of a moving synthetic object.
Motion direction, displacement, and pixel noise are generated deterministically, so the task is
fully paired and offline.

## Implementation

- `data.py` returns two endpoint frames and the true midpoint.
- `model.py` concatenates both endpoints and uses residual convolutional blocks to predict `mid`.
- `train.py` optimizes L1 interpolation loss and reports validation PSNR.

## Quick Run

```bash
python -m tracks.vision.lesson_68_synthetic_video_frame_interpolation.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_68_synthetic_video_frame_interpolation/<run_name>/`. Success
requires the standard config, metrics, log, and checkpoint artifacts, finite train/eval L1 loss,
and a finite `eval_psnr` value.
