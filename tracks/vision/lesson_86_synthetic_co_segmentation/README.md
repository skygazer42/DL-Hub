# Lesson 86: Synthetic Co-Segmentation

This lesson segments the object shared across a group of RGB images while ignoring independently
sampled distractors. The shared shape and color persist across the set with small spatial shifts,
providing group-level supervision entirely from the local renderer.

## Implementation

- `data.py` returns image groups with binary masks and foreground/background class indices.
- `model.py` wraps the local `coseg:siamese_coseg_small` Zoo model and exposes logits, masks, group
  tokens, and a match map.
- `train.py` combines pixel cross-entropy with Dice loss and reports group-mask IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_86_synthetic_co_segmentation.train \
  --epochs 1 --num-samples 48 --batch-size 4 --set-size 3 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_86_synthetic_co_segmentation/<run_name>/`. Success requires
the standard config, metrics, log, and checkpoint artifacts, finite cross-entropy/Dice losses, and
train/eval IoU values in `[0, 1]`.
