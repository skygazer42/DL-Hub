# Lesson 81: Synthetic Shadow Detection

This lesson introduces a toy paired setup for shadow detection and relighting:

- render clean RGB scenes with simple geometric content
- synthesize soft shadow masks and boundary cues
- darken masked regions to produce shadowed observations
- train a tiny shadow detector with an auxiliary relighting head

The setup is fully synthetic and CPU-friendly, so it runs quickly without external datasets.

## Files

- `data.py`: synthetic clean/shadowed image generation and dataloaders
- `model.py`: tiny `dlhub`-backed shadow detector + relighting head
- `train.py`: supervised training loop with mask/boundary/relighting losses

## Smoke Test

```bash
python -m tracks.vision.lesson_81_synthetic_shadow_detection.train \
  --epochs 1 \
  --num-samples 48 \
  --batch-size 4 \
  --image-size 32 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs are written to `outputs/vision/lesson_81_synthetic_shadow_detection/<run_name>/`.
