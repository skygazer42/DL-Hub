# Lesson 29: Compact Diffusion Multi-Reference Generation

This lesson demonstrates a compact diffusion setup where a target image is generated from two
reference appearance images plus a condition map that describes where content should appear.

The lesson covers:
- deterministic synthetic `(reference_a, reference_b, condition, target)` data
- a tiny conditional diffusion denoiser with multi-reference conditioning
- saved sample tensors, denoising trajectories, and checkpoints

Run locally with:

```bash
python -m tracks.generative.lesson_29_compact_diffusion_multi_reference_generation.train --device cpu --epochs 1
```
