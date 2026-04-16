# Lesson 11: Toy ControlNet

This lesson demonstrates a tiny diffusion denoiser with a side conditioning branch over
structural guidance (a simple edge-like hint map). The design mirrors ControlNet ideas
at toy scale: the main branch predicts diffusion noise from `x_t`, while the guidance
branch injects residual features derived from a structure hint.

The setup is CPU-friendly and synthetic: generated grayscale shapes, lightweight CNN
blocks, and standard DDPM-style noising/denoising.

## Run

```bash
python -m tracks.generative.lesson_11_toy_controlnet.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_11_toy_controlnet -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_11_toy_controlnet/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
