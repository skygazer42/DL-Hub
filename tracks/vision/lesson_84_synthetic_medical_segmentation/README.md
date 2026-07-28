# Lesson 84: Synthetic Medical Segmentation

This lesson builds a compact-first medical segmentation pipeline:

- generate synthetic grayscale slices with tissue and lesion regions
- train a tiny medical segmenter from `dlhub.vision.medical_segmentation`
- optimize multiclass segmentation with cross-entropy and track Dice score

The setup is CPU-friendly and intended for teaching and smoke-test workflows.

## Run

```bash
python -m tracks.vision.lesson_84_synthetic_medical_segmentation.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu \
  --backbone-family unet \
  --backbone-variant unet_tiny
```

