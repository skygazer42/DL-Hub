# Lesson 39: Synthetic Face Alignment

This lesson frames face alignment as a compact regression task over synthetic face crops. Each
input image contains a face rendered with small rotation, scale, and translation perturbations,
while the target is a canonical five-point facial geometry layout.

## What It Teaches

- generating paired posed-face inputs and canonical alignment targets
- lightweight regression for normalized aligned landmark prediction
- pixel-space alignment error tracking for a small face-geometry lesson

## Run

```bash
python -m tracks.vision.lesson_39_synthetic_face_alignment.train --device cpu --epochs 1
```
