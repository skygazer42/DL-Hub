# Lesson 30: Synthetic Salient-Object Box Regression

This lesson localizes the highest-intensity circle or rectangle among weaker distractors. The target
mask is converted to a normalized center/width/height box, turning the preceding salient-mask task
into compact single-object detection.

## Implementation

- `data.py` renders the scene and returns one normalized `cx, cy, width, height` target.
- `model.py` pools residual CNN features and regresses four bounded box values.
- `train.py` combines L1 and IoU losses and reports mean box IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_30_synthetic_salient_object_detection_boxes/<run_name>/`.
Success requires `config.json`, `metrics.jsonl`, `logs/train.log`, and a checkpoint, with finite L1
and IoU losses and train/eval IoU values in `[0, 1]`.
