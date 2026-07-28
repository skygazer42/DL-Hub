# Lesson 16 — Translation Style Transfer (CycleGAN-style, compact-first)

This lesson trains a tiny CycleGAN-style model on synthetic unpaired domains:
- Domain A: noisy squares
- Domain B: stripe patterns

It is designed to be CPU-friendly and self-contained (no datasets to download).

## Run

```bash
python -m tracks.vision.lesson_16_style_transfer_translation_cyclegan.train \
  --epochs 1 --max-train-batches 2 \
  --image-size 32 --batch-size 4 \
  --device cpu --run-name dev
```

Outputs land in:

`outputs/vision/lesson_16_style_transfer_translation_cyclegan/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt` (a/b/fake/recon tensors)
- `checkpoints/checkpoint.pt`

