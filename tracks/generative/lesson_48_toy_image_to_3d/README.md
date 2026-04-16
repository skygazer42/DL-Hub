# Lesson 48: Toy Image-to-3D

This lesson demonstrates a tiny, CPU-friendly image-to-3D training loop built on
`dlhub.generative.image_to_3d` builders.

## What It Covers

- synthetic RGB image generation
- lightweight pseudo-3D supervision:
  - volumetric density target
  - mesh token target
- wrapping a `dlhub` image-to-3D family as a training model
- one-file training loop with checkpoint + metrics outputs

## Run

```bash
python -m tracks.generative.lesson_48_toy_image_to_3d.train \
  --epochs 1 \
  --device cpu \
  --family zero123_toy \
  --variant zero123_toy_tiny \
  --run-name dev
```

## Outputs

The run writes to:

- `outputs/generative/lesson_48_toy_image_to_3d/<run_name>/config.json`
- `outputs/generative/lesson_48_toy_image_to_3d/<run_name>/metrics.jsonl`
- `outputs/generative/lesson_48_toy_image_to_3d/<run_name>/samples.pt`
- `outputs/generative/lesson_48_toy_image_to_3d/<run_name>/checkpoints/checkpoint.pt`
