# Lesson 22: Synthetic Road Scene Understanding

This lesson extends the lane-focused Batch 18 lessons into a compact scene-understanding task:

- render a compact road image with lane slots, simple objects, and scene cues
- predict which lane slots are available
- predict which object types are present
- classify the overall road-scene pattern

The implementation stays compact-first and CPU-friendly so the whole loop is easy to inspect.

## Files

- `data.py`: deterministic synthetic road-scene renderer and dataloaders
- `model.py`: tiny encoder with lane/object query heads plus a fused scene classifier
- `train.py`: supervised training loop with multilabel + scene classification losses

## Smoke Test

```bash
python -m tracks.vision.lesson_22_synthetic_road_scene_understanding.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --image-size 48 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_22_synthetic_road_scene_understanding/<run_name>/`.
