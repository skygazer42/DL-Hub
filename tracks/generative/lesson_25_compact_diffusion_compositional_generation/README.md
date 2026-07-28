# Lesson 25: Compact Diffusion Compositional Generation

This lesson demonstrates a compact diffusion denoiser conditioned on two complementary signals:
`structure` provides where content should appear, and `style` provides the texture that should be
composed into the final image.

Each sample is CPU-friendly: synthetic 28x28 grayscale data, lightweight conditioning maps, and a
small CNN denoiser.

## Run

```bash
python -m tracks.generative.lesson_25_compact_diffusion_compositional_generation.train --epochs 1 --device cpu
```
