# Lesson 66: Synthetic Video Object Detection

This lesson predicts fixed-slot object presence, class, and normalized boxes from a short grayscale
clip. One or two rectangular objects move with slot-specific velocities; targets describe the
active slots and their starting boxes.

## Implementation

- `data.py` returns clips with `boxes`, `labels`, and `present` target tensors.
- `model.py` encodes spatiotemporal content and emits per-slot box, score, and class predictions.
- `train.py` combines box, presence-score, and class losses.

## Quick Run

```bash
python -m tracks.vision.lesson_66_synthetic_video_object_detection.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_66_synthetic_video_object_detection/<run_name>/`. Success
requires the standard config, metrics, log, and checkpoint artifacts, with finite `box_loss`,
`score_loss`, and `class_loss` for training and evaluation.
