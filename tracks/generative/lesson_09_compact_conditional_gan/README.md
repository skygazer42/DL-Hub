# Lesson 09: Compact Conditional GAN

This lesson trains a class-conditional GAN on a deterministic synthetic image task. Four labels
select vertical, horizontal, main-diagonal, or anti-diagonal patterns, so the complete pipeline is
offline and small enough for a CPU smoke run.

## Implementation

- `data.py` renders labeled `28 x 28` grayscale patterns and creates a seeded train/validation split.
- `model.py` defines MLP generator and discriminator branches. Both branches receive learned label
  embeddings; the generator combines them with latent noise.
- `train.py` alternates discriminator and generator BCE updates and records `d_loss` and `g_loss`.

## Quick Run

```bash
python -m tracks.generative.lesson_09_compact_conditional_gan.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 2 \
  --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/generative/lesson_09_compact_conditional_gan/<run_name>/`. A
successful run exits with code 0 and writes `config.json`, `metrics.jsonl`, `samples.pt`,
`logs/train.log`, and `checkpoints/checkpoint.pt`. The metrics file must contain finite generator
and discriminator train/validation losses, while `samples.pt` stores generated images with their
conditioning labels.
