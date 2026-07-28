# Lesson 22: Compact Diffusion Style Transfer

This lesson demonstrates a tiny conditional diffusion denoiser for style transfer using synthetic
triples of `(content, style, stylized target)` grayscale images. The model predicts diffusion noise
for a noised stylized target while conditioning on both the source content image and a style
texture reference.

## Run

```bash
python -m tracks.generative.lesson_22_compact_diffusion_style_transfer.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs

`outputs/generative/lesson_22_compact_diffusion_style_transfer/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `style_trajectory.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
