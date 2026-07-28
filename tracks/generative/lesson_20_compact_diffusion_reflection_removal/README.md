# Lesson 20: Compact Diffusion Reflection Removal

This lesson demonstrates a tiny conditional diffusion denoiser for reflection
removal using synthetic paired `(reflected, clean)` grayscale images. The model
predicts diffusion noise for a noised clean target while conditioning on a
reflection-corrupted observation.

The setup is CPU-friendly: synthetic 28x28 shapes, lightweight reflection
overlay synthesis, and a compact CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_20_compact_diffusion_reflection_removal.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_20_compact_diffusion_reflection_removal -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_20_compact_diffusion_reflection_removal/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
