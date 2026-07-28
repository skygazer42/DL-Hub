# Lesson 88: Synthetic Person ReID

This lesson introduces a compact-first person re-identification pipeline:

- generate synthetic person-like images with identity-specific appearance cues
- train a compact re-id model from `dlhub.vision.reid` (default: `osnet_tiny`)
- optimize identity classification plus embedding triplet separation

The setup stays CPU-friendly and smoke-testable for quick experimentation.

## Run

```bash
python -m tracks.vision.lesson_88_synthetic_reid.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

