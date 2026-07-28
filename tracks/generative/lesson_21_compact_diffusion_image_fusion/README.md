# Lesson 21: Compact Diffusion Image Fusion

This lesson demonstrates a tiny conditional diffusion denoiser for fusing two
synthetic complementary observations into a clean target image. Each sample is
a grayscale shape with paired views `(obs_a, obs_b)` that expose different
parts of the same underlying signal.

The setup is CPU-friendly: synthetic 28x28 data, lightweight corruption, and a
compact CNN denoiser conditioned on both observations and timestep.

## Run

```bash
python -m tracks.generative.lesson_21_compact_diffusion_image_fusion.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_21_compact_diffusion_image_fusion -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_21_compact_diffusion_image_fusion/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
