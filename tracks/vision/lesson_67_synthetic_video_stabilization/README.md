# Lesson 67: Synthetic Video Stabilization

This lesson reconstructs a stable moving-blob sequence from a jittered, blurred, and noisy clip.
Each frame receives an independent integer translation bounded by `--max-jitter`, while the clean
trajectory is retained as the target.

## Implementation

- `data.py` returns jittered clips paired with their stabilized sequences.
- `model.py` predicts a stabilized frame sequence with a compact video restoration network.
- `train.py` minimizes reconstruction loss on all frames.

## Quick Run

```bash
python -m tracks.vision.lesson_67_synthetic_video_stabilization.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_67_synthetic_video_stabilization/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and a checkpoint, with finite training and
evaluation reconstruction losses.
