# Lesson 15: Toy Diffusion Super Resolution

This lesson demonstrates a tiny diffusion-style super-resolution setup with paired
synthetic low-resolution and high-resolution images.

- Data: grayscale toy scenes (simple geometric shapes), degraded to low-resolution by
  downsampling.
- Model: a small diffusion denoiser conditioned on the low-resolution image (upsampled
  inside the model).
- Goal: generate high-resolution samples that match the structural guidance of the
  paired low-resolution input.

## Run

```bash
python -m tracks.generative.lesson_15_toy_diffusion_super_resolution.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_15_toy_diffusion_super_resolution -- --epochs 1 --device cpu
```
