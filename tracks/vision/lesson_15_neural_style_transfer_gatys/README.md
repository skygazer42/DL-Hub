# Lesson 15 — Neural Style Transfer (Gatys-style, toy-first)

This lesson runs a tiny optimization-based neural style transfer loop.

It is designed to be CPU-friendly and self-contained (no datasets to download).

## Run

```bash
python -m tracks.vision.lesson_15_neural_style_transfer_gatys.train \
  --steps 8 --image-size 64 --batch-size 2 \
  --device cpu --run-name dev
```

Outputs land in:

`outputs/vision/lesson_15_neural_style_transfer_gatys/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `stylized.pt` (content/style/stylized tensors)
- `stylized.png` (optional, if torchvision is installed)

