# Lesson 23: Synthetic Image Dehazing

This lesson adds a compact-first paired dehazing loop to the vision track:

- render a clean synthetic image with simple geometric content
- build a transmission map from a compact depth prior
- mix the clean image with atmospheric light to create haze
- train a small model to predict both the restored image and the transmission map

The setup is CPU-friendly and fully synthetic, so it runs quickly without external datasets.

## Files

- `data.py`: clean-image renderer, haze synthesis, and dataloaders
- `model.py`: tiny residual dehazing network with restoration and transmission heads
- `train.py`: supervised training loop with paired reconstruction metrics

## Smoke Test

```bash
python -m tracks.vision.lesson_23_synthetic_image_dehazing.train \
  --epochs 1 \
  --num-samples 48 \
  --batch-size 4 \
  --image-size 32 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_23_synthetic_image_dehazing/<run_name>/`.
