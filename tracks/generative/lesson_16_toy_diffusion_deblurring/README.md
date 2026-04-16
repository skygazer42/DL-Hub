# Lesson 16: Toy Diffusion Deblurring

This lesson demonstrates a tiny diffusion-style denoiser for image deblurring using
paired synthetic `(blurry, sharp)` grayscale samples. The model predicts diffusion
noise for a noised sharp target while conditioning on the blurry observation.

The setup is CPU-friendly: synthetic 28x28 shapes, deterministic blur kernel, and a
small CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_16_toy_diffusion_deblurring.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_16_toy_diffusion_deblurring -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_16_toy_diffusion_deblurring/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
