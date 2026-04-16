# Lesson 10: Toy Diffusion Image Editing

This lesson demonstrates diffusion-style image editing with three conditioning signals:

- `source` image (the image to edit)
- binary `mask` (where edits are allowed)
- `control_token` (which edit operation to apply)

The setup is toy-first and CPU-friendly: synthetic grayscale shapes, a tiny convolutional
denoiser, and masked noise prediction. During sampling, pixels outside the edit mask are
kept equal to the source image.

## Run

```bash
python -m tracks.generative.lesson_10_toy_diffusion_image_editing.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

Or via the shared launcher:

```bash
python scripts/run_lesson.py generative lesson_10_toy_diffusion_image_editing -- --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_10_toy_diffusion_image_editing/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `edited_samples.pt`
- `denoise_grid.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
