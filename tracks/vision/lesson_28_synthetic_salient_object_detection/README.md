# Lesson 28: Synthetic Salient-Object Detection

This lesson segments the brightest object among several lower-contrast distractors. Each grayscale
scene contains random circles or rectangles over noisy background, with a binary mask only for the
salient shape.

## Implementation

- `data.py` controls object sizes, distractor counts, and noise through `DataConfig`.
- `model.py` predicts a dense salient-mask logit map with a residual CNN.
- `train.py` combines BCE and Dice losses and reports mask IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_28_synthetic_salient_object_detection.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_28_synthetic_salient_object_detection/<run_name>/`. Success
requires `config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`, finite
BCE/Dice losses, and train/eval IoU values in `[0, 1]`.
