# Lesson 24: Synthetic Reflection Removal

This lesson introduces a compact paired reflection-removal setup:

- render a clean transmission image with simple geometric content
- render a secondary reflection layer and blur it to mimic glass reflections
- blend transmission and reflection into a single observed mixture
- train a tiny dual-head model to recover both transmission and reflection

The pipeline is fully synthetic and CPU-friendly, so it runs quickly without external datasets.

## Files

- `data.py`: synthetic transmission/reflection generation and dataloaders
- `model.py`: tiny residual network with transmission and reflection heads
- `train.py`: supervised training loop with separated reconstruction metrics

## Smoke Test

```bash
python -m tracks.vision.lesson_24_synthetic_reflection_removal.train \
  --epochs 1 \
  --num-samples 48 \
  --batch-size 4 \
  --image-size 32 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_24_synthetic_reflection_removal/<run_name>/`.
