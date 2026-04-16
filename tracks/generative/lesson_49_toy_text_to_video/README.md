# Lesson 49: Toy Text-to-Video

This lesson is a minimal, CPU-friendly text-to-video pipeline.

## What you build

- a synthetic prompt/video dataset (`8x8`, short clips)
- a tiny text-to-video model wrapper around `dlhub.generative.text_to_video`
- a simple supervised training loop with reproducible outputs

## Run

```bash
python -m tracks.generative.lesson_49_toy_text_to_video.train \
  --epochs 3 \
  --batch-size 8 \
  --frames 4 \
  --family diffusion_t2v \
  --variant diffusion_t2v_tiny
```

## Outputs

Runs are stored under:

- `outputs/generative/lesson_49_toy_text_to_video/<run_name>/config.json`
- `outputs/generative/lesson_49_toy_text_to_video/<run_name>/metrics.jsonl`
- `outputs/generative/lesson_49_toy_text_to_video/<run_name>/samples.pt`
- `outputs/generative/lesson_49_toy_text_to_video/<run_name>/checkpoints/checkpoint.pt`
