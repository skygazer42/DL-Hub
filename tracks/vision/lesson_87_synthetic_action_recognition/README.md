# Lesson 87: Synthetic Action Recognition

This lesson classifies four motion trajectories of a bright spot: two horizontal paths at different
heights, a circular orbit, and a rapid horizontal oscillation. Seeded position jitter and image
noise make the offline clips non-identical while preserving their action labels.

## Implementation

- `data.py` renders eight-frame grayscale clips and returns `action_label`.
- `model.py` wraps the local configurable C3D video-classifier family.
- `train.py` optimizes action cross-entropy and records clip-level accuracy.

## Quick Run

```bash
python -m tracks.vision.lesson_87_synthetic_action_recognition.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_87_synthetic_action_recognition/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`, with finite action
loss and train/eval accuracy values in `[0, 1]`.
