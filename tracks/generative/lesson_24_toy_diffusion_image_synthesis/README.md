# Lesson 24: Toy Diffusion Image Synthesis

This lesson demonstrates a tiny conditional diffusion denoiser for synthetic image synthesis using
pairs of `(condition, target)` grayscale images. The model predicts diffusion noise for a noised
target image while conditioning on a coarse structural condition map.

## Run

```bash
python -m tracks.generative.lesson_24_toy_diffusion_image_synthesis.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs

`outputs/generative/lesson_24_toy_diffusion_image_synthesis/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `trajectory.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
