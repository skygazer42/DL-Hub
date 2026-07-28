# Lesson 78: Synthetic Image Relighting

This lesson is a compact-first relighting task for CPU-friendly training loops.

## Task

- Input: source image `x` with shape `(3, H, W)`.
- Targets:
- relit image `y_relit` with shape `(3, H, W)`.
- illumination map `y_light` with shape `(1, H, W)`.

## Model

`RelightingModel` wraps an existing relighter builder from `dlhub.vision.image_relighting`
via `arch="<family>:<variant>"`, for example `deep_relight:deep_relight_tiny`.

## Loss

`loss = L1(relit, target_relit) + 0.2 * L1(light_map, target_light_map)`

## Smoke Run

```bash
python -m tracks.vision.lesson_78_synthetic_image_relighting.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --arch deep_relight:deep_relight_tiny \
  --run-name smoke
```
