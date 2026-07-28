# Lesson 08: Compact Diffusion Transformer (DiT-style)

This lesson is a small, teaching-first bridge from compact diffusion to transformer denoisers:

- keep a standard DDPM noise-prediction objective
- replace the MLP/CNN denoiser with a tiny patch transformer
- train on synthetic grayscale shapes so CPU smoke runs stay fast and offline

The model is intentionally tiny: patch embedding, a short transformer encoder stack,
and patch reconstruction back to image space.

## Run

```bash
python -m tracks.generative.lesson_08_compact_diffusion_transformer.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or through the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_08_compact_diffusion_transformer -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_08_compact_diffusion_transformer/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
