# Lesson 26: Synthetic Text Detection

This lesson detects one text-like striped patch in a small RGB scene. A configurable fraction of
images is empty, so the model must jointly predict whether text is present and regress its normalized
`xyxy` bounding box. The renderer is deterministic and offline.

## Implementation

- `data.py` returns an image with `bbox` and binary `score` targets.
- `model.py` uses a residual CNN with separate box and presence-score heads.
- `train.py` combines box regression with binary score loss and reports IoU plus score accuracy.

## Quick Run

```bash
python -m tracks.vision.lesson_26_synthetic_text_detection.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_26_synthetic_text_detection/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`. The metrics must
contain finite box/score losses, IoU in `[0, 1]`, and score accuracy in `[0, 1]`.
