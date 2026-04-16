# Lesson 79: Synthetic Transparent Object Segmentation

This lesson builds a toy transparent-object segmentation pipeline with fully synthetic images:

- render a smooth RGB background
- place one synthetic transparent object as an ellipse mask
- blend the object with per-sample alpha to form the input image
- supervise three targets: object mask, alpha map, and mask boundary
- train a tiny segmenter from `dlhub.vision.transparent_object_segmentation`

The setup is intentionally lightweight and CPU-friendly.

## Files

- `data.py`: synthetic transparent-scene generation and dataloaders
- `model.py`: wrapper over `dlhub` transparent-segmentation model builders
- `train.py`: supervised training/evaluation loop with IoU reporting

## Smoke Test

```bash
python -m tracks.vision.lesson_79_synthetic_transparent_object_segmentation.train \
  --epochs 1 \
  --num-samples 48 \
  --batch-size 4 \
  --image-size 32 \
  --arch glassseg_toy \
  --variant glassseg_toy_tiny \
  --width-mult 0.75 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_79_synthetic_transparent_object_segmentation/<run_name>/`.
