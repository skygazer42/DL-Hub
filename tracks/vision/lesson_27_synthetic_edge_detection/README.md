# Lesson 27: Synthetic Edge Detection

This lesson predicts object boundaries in grayscale scenes containing random rectangles and circles.
Targets are computed directly from horizontal and vertical mask differences, so no external image
or annotation source is required.

## Implementation

- `data.py` renders bright shapes over noise and derives a one-pixel edge mask.
- `model.py` applies a compact residual CNN to produce edge logits at input resolution.
- `train.py` combines BCE and Dice losses and reports thresholded edge IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_27_synthetic_edge_detection.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_27_synthetic_edge_detection/<run_name>/`. Success requires
`config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`, with finite BCE and
Dice losses and train/eval IoU values in `[0, 1]`.
