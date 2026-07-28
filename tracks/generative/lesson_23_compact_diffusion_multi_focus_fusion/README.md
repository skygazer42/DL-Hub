# Lesson 23: Compact Diffusion Multi-Focus Fusion

This lesson demonstrates a tiny conditional diffusion denoiser for fusing two synthetic
multi-focus observations into a clean target image. Each sample is a grayscale shape with paired
views `(focus_a, focus_b)` where different spatial regions remain sharp in each input.

The setup stays CPU-friendly: synthetic 28x28 data, lightweight blur corruption, and a compact
CNN denoiser conditioned on both focus planes and timestep.

## Run

```bash
python -m tracks.generative.lesson_23_compact_diffusion_multi_focus_fusion.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_23_compact_diffusion_multi_focus_fusion -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_23_compact_diffusion_multi_focus_fusion/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `focus_trajectory.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
