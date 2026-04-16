# Lesson 19: Toy Diffusion Dehazing (去雾)

This lesson demonstrates a tiny conditional diffusion setup for synthetic image
dehazing using paired `(hazy, clean)` grayscale samples. The model predicts
diffusion noise for a noised clean target while conditioning on the hazy input.

The setup is CPU-friendly: synthetic 28x28 shapes, lightweight atmospheric haze
corruption, and a small CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_19_toy_diffusion_dehazing.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_19_toy_diffusion_dehazing -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_19_toy_diffusion_dehazing/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
