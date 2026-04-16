# Lesson 18: Toy Diffusion Deraining (去雨)

This lesson demonstrates a tiny conditional diffusion setup for synthetic image
deraining using paired `(rainy, clean)` grayscale samples. The model predicts
diffusion noise for a noised clean target while conditioning on the rainy input.

The setup is CPU-friendly: synthetic 28x28 shapes, lightweight rain streak
corruption, and a small CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_18_toy_diffusion_deraining.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_18_toy_diffusion_deraining -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_18_toy_diffusion_deraining/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
