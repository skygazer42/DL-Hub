# Lesson 74: Synthetic Video Instance Segmentation

This lesson predicts separate per-frame masks for fixed object slots. Multiple synthetic instances
move along distinct trajectories, and the target channel index preserves each identity through the
clip; it is a fixed-slot teaching setup rather than proposal-based instance detection.

## Implementation

- `data.py` returns a clip with `instance_masks` shaped `(time, instances, height, width)`.
- `model.py` produces the same number of instance-logit channels at every frame.
- `train.py` optimizes mask BCE across all slots and timesteps.

## Quick Run

```bash
python -m tracks.vision.lesson_74_synthetic_video_instance_segmentation.train \
  --epochs 1 --num-samples 48 --batch-size 4 --num-instances 2 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_74_synthetic_video_instance_segmentation/<run_name>/`. Success
requires `config.json`, `metrics.jsonl`, `logs/train.log`, and a checkpoint, with finite training
and evaluation `mask_bce_loss` values.
