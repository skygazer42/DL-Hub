# Lesson 26: Compact Diffusion Image Variation

This lesson demonstrates a small conditional diffusion denoiser for image variation.
Each sample contains a `(source, target)` pair where `source` is a perturbed view of
the same synthetic object found in `target` (shifted, blurred, and noised). The model
predicts diffusion noise for a noised `target` while conditioning on `source`.

## Run

```bash
python -m tracks.generative.lesson_26_compact_diffusion_image_variation.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs

`outputs/generative/lesson_26_compact_diffusion_image_variation/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `trajectory.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
