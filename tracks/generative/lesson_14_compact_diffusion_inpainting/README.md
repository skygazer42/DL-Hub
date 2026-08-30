# Lesson 14: Compact Diffusion Inpainting

This lesson reconstructs randomly masked regions of synthetic circles, rectangles, and crosses.
The dataset returns the visible context, full target, and binary edit mask; all samples are generated
deterministically and require no network access.

## Implementation

- `data.py` paints a shape, removes a random rectangular region, and returns context/target/mask triples.
- `model.py` conditions a compact timestep-aware convolutional denoiser on both context and mask.
- `train.py` optimizes noise prediction only inside the edit mask and tracks masked noise MSE.

## Quick Run

```bash
python -m tracks.generative.lesson_14_compact_diffusion_inpainting.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/generative/lesson_14_compact_diffusion_inpainting/<run_name>/`.
A successful run writes `config.json`, `metrics.jsonl`, `samples.pt`, `denoise_grid.pt`,
`logs/train.log`, and `checkpoints/checkpoint.pt`. The metrics must include finite
`train_masked_noise_mse` and `val_masked_noise_mse`; `samples.pt` contains context, target, mask,
and completed samples for inspection.
