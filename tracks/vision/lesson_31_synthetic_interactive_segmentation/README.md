# Lesson 31: Synthetic Interactive Segmentation

This lesson segments one target shape from a cluttered grayscale image using a positive click.
The click is rendered as a small binary disk inside the target while other shapes act as distractors;
all images, clicks, and masks are generated locally.

## Implementation

- `data.py` returns image, click map, and target mask triples.
- `model.py` concatenates the image and click map and predicts a dense mask with a residual CNN.
- `train.py` combines BCE and Dice losses and reports target-mask IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_31_synthetic_interactive_segmentation.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_31_synthetic_interactive_segmentation/<run_name>/`. Success
requires `config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`, finite
BCE/Dice losses, and train/eval IoU values in `[0, 1]`.
