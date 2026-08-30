# Lesson 70: Synthetic Video Understanding

This lesson classifies four temporal events: horizontal motion, vertical motion, diagonal motion,
and a stationary blinking spot. Short grayscale clips include a smooth background and configurable
noise, with no external video dependencies.

## Implementation

- `data.py` renders one event and returns its `event_label`.
- `model.py` uses compact 3D convolution blocks and a clip-level event head.
- `train.py` optimizes event cross-entropy and records classification accuracy.

## Quick Run

```bash
python -m tracks.vision.lesson_70_synthetic_video_understanding.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_70_synthetic_video_understanding/<run_name>/`. Success requires
the standard config, metrics, log, and checkpoint artifacts, finite event loss, and train/eval
accuracy values in `[0, 1]`.
