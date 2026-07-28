# Lesson 27: Compact Diffusion Reference-Guided Generation

This lesson demonstrates a compact diffusion setup where a target image is generated from a
reference appearance image and a separate condition map that describes where content should
appear.

The lesson covers:
- deterministic synthetic `(reference, condition, target)` data
- a tiny conditional diffusion denoiser
- saved sample tensors, denoising trajectories, and checkpoints

Run locally with:

```bash
python -m tracks.generative.lesson_27_compact_diffusion_reference_guided_generation.train --device cpu --epochs 1
```
