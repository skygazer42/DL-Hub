# Lesson 05: Compact Consistency Model (one-step generation)

This lesson trains a consistency model (Song et al., 2023) in its
simplest compact form:

- a consistency function `f(x, sigma)` with the skip parameterization
  `c_skip(sigma) * x + c_out(sigma) * F(x, sigma)`, so the boundary
  condition `f(x, sigma_min) = x` holds by construction
- consistency training (no pre-trained diffusion teacher): adjacent
  noise levels on a Karras sigma grid share one noise draw, and the
  online network at the higher level is pulled toward an EMA target
  network at the lower level
- one-step sampling `f(sigma_max * z, sigma_max)`, plus optional
  multistep stochastic refinement
- default to synthetic MNIST-like blobs so the lesson runs offline

## Run

Offline smoke run:

```bash
python -m tracks.generative.lesson_05_compact_consistency_model.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

If `torchvision` is installed, you can switch to real MNIST:

```bash
python -m tracks.generative.lesson_05_compact_consistency_model.train --dataset mnist --epochs 5
```

Useful knobs: `--num-discretization-steps` (sigma grid size),
`--sigma-min/--sigma-max/--sigma-data`, `--ema-decay`,
`--num-sample-steps` (1 = one-step generation).

## Outputs

`outputs/generative/lesson_05_compact_consistency_model/<run_name>/`

- `config.json`
- `metrics.jsonl` (`train_consistency_mse` / `val_consistency_mse`)
- `samples.pt` (one-step samples from the EMA target network)
- `refine_grid.pt` (multistep refinement frames)
- `checkpoints/checkpoint.pt` (online network + EMA target network)

## Exercises

1. Set `--num-sample-steps 4` and compare `refine_grid.pt` frames with
   the one-step samples — how much does refinement sharpen the blobs?
2. Lower `--ema-decay` (e.g. 0.9) and watch `train_consistency_mse`:
   why does a faster-moving target make the loss noisier?
3. Increase `--num-discretization-steps` and check whether one-step
   samples improve. What does a finer grid change about the target the
   student is chasing?
