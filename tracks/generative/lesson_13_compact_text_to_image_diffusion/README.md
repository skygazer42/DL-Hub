# Lesson 13: Compact Text-to-Image Diffusion

This lesson conditions a small diffusion denoiser on one of four prompt tokens: vertical bar,
horizontal bar, square blob, or ring blob. Images and prompts are generated locally, keeping both
training and reverse-diffusion sampling independent of external datasets or model downloads.

## Implementation

- `data.py` renders prompt-aligned `28 x 28` grayscale shapes and seeded train/validation loaders.
- `model.py` combines timestep embeddings and a learned prompt embedding in a convolutional noise
  predictor, with a configurable linear beta schedule.
- `train.py` learns the DDPM noise target and logs training and validation noise MSE.

## Quick Run

```bash
python -m tracks.generative.lesson_13_compact_text_to_image_diffusion.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/generative/lesson_13_compact_text_to_image_diffusion/<run_name>/`.
A successful run writes `config.json`, `metrics.jsonl`, `samples.pt`, `denoise_grid.pt`,
`logs/train.log`, and `checkpoints/checkpoint.pt`. Acceptance requires finite `train_noise_mse` and
`val_noise_mse` entries plus sampled tensors paired with their `token_ids`.
