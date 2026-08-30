# Lesson 21: Compact Adversarial-Example Detection

This lesson predicts whether a synthetic sentence is clean or token-replaced. Clean examples use a
coherent keyword/context/action triple; adversarial examples swap that triple for another intent.
`--adversarial-fraction` controls the class balance.

## Implementation

- `data.py` creates labeled clean and attacked sentences, builds a vocabulary, and performs a seeded split.
- `model.py` mean-pools learned token embeddings and produces one binary detection logit.
- `train.py` optimizes binary cross-entropy and records detector accuracy.

## Quick Run

```bash
python -m tracks.nlp.lesson_21_compact_adversarial_example_detection.train \
  --epochs 1 --num-samples 128 --batch-size 16 --adversarial-fraction 0.5 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_21_compact_adversarial_example_detection/<run_name>/`.
A successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite train/eval losses and `train_acc` plus
`eval_acc` values in `[0, 1]`.
