# Lesson 20: Synthetic Lane Detection

This lesson introduces a toy-first lane detection loop with dense supervision:

- render a simple road image with several curved lane centerlines
- predict a lane heatmap that highlights likely lane pixels
- regress the normalized x-coordinate of the lane centerline where lanes are visible

The implementation is intentionally small so the full data/model/train loop remains easy to inspect.

## Files

- `data.py`: deterministic synthetic lane renderer and dataloaders
- `model.py`: tiny encoder-decoder with two prediction heads
- `train.py`: supervised training loop with heatmap + offset losses

## Smoke Test

```bash
python -m tracks.vision.lesson_20_synthetic_lane_detection.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --image-size 48 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_20_synthetic_lane_detection/<run_name>/`.
