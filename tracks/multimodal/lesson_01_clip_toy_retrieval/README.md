# Lesson 01: CLIP-Style Toy Retrieval

This lesson teaches the smallest runnable image-text alignment loop:

- render synthetic images from simple attributes
- encode image and text separately
- project both into one shared embedding space
- train with a symmetric contrastive loss
- inspect retrieval accuracy directly

## Run

```bash
python -m tracks.multimodal.lesson_01_clip_toy_retrieval.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_01_clip_toy_retrieval/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Replace mean-pooled text features with a tiny GRU encoder.
2. Add harder negatives by allowing two captions to share color and shape.
3. Visualize the learned embedding space with two-dimensional projections.
