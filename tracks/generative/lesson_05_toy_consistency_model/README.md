# Lesson 03: Toy Diffusion (MNIST-like, minimal DDPM)

This lesson keeps diffusion toy-first:

- train a tiny denoiser to predict added Gaussian noise
- use a small linear noise schedule so CPU smoke runs stay fast
- default to synthetic MNIST-like blobs so the lesson runs offline

## Run

Offline smoke run:

```bash
python -m tracks.generative.lesson_03_toy_diffusion_mnist.train --dataset fake --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

If `torchvision` is installed, you can switch to real MNIST:

```bash
python -m tracks.generative.lesson_03_toy_diffusion_mnist.train --dataset mnist --epochs 5
```

## Outputs

`outputs/generative/lesson_03_toy_diffusion_mnist/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

If `torchvision` is available, the lesson also writes `samples.png` and `denoise_grid.png`.

## Exercises

1. Increase `--num-diffusion-steps` and compare sample smoothness versus speed.
2. Swap the MLP denoiser for a tiny conv net that keeps the same training target.
3. Try a cosine-style schedule after understanding the linear baseline here.
