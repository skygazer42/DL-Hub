# Lesson 46: Compact Image-to-Video Diffusion

This lesson generates a short video from one source image. Synthetic textured shapes are translated
over time with a small brightness pulse and noise, providing paired source images and target clips
without external video assets.

## Implementation

- `data.py` renders a source shape and a configurable sequence of motion frames.
- `model.py` combines source-image conditioning, diffusion timestep embeddings, and frame-position
  embeddings in a compact video noise predictor.
- `train.py` learns the video noise target and records training and validation noise MSE.

## Quick Run

```bash
python -m tracks.generative.lesson_46_compact_image_to_video_diffusion.train \
  --epochs 1 --num-samples 64 --batch-size 8 --num-frames 4 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/generative/lesson_46_compact_image_to_video_diffusion/<run_name>/`.
A successful run writes `config.json`, `metrics.jsonl`, `samples.pt`, `trajectory.pt`,
`logs/train.log`, and `checkpoints/checkpoint.pt`. Acceptance requires finite `train_noise_mse` and
`val_noise_mse` records; sample artifacts retain the source, target video, generated clip, and
denoising trajectory.
