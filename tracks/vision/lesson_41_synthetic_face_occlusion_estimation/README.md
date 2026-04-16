# Lesson 41: Synthetic Face Occlusion Estimation

This lesson frames face occlusion estimation as a compact regression task over synthetic face
crops. Each image contains a simple face rendering plus an overlaid occluder, and the target is
the fraction of face pixels covered by that occluder.

## What It Teaches

- generating paired face crops and scalar occlusion-ratio targets
- lightweight regression over synthetic face observations
- mean absolute error tracking for a compact vision regression lesson

## Run

```bash
python -m tracks.vision.lesson_41_synthetic_face_occlusion_estimation.train --device cpu --epochs 1
```
