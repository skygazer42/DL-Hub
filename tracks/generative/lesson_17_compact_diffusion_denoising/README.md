# Lesson 17: Compact Diffusion Denoising (去噪)

This lesson demonstrates a tiny conditional diffusion denoiser for synthetic
paired `(noisy, clean)` grayscale images. The model predicts diffusion noise for
a noised clean target while conditioning on the observed noisy input.

The setup is CPU-friendly: synthetic 28x28 shapes, lightweight Gaussian plus
impulse corruption, and a compact CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_17_compact_diffusion_denoising.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_17_compact_diffusion_denoising -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_17_compact_diffusion_denoising/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
