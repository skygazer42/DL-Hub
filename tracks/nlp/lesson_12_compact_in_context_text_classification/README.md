# Lesson 12: Compact In-Context Text Classification

This lesson demonstrates in-context text classification without gradient updates.

Each sample contains:
- support examples with labels (inside a prompt-like context block)
- one query text
- a classifier that predicts by comparing query tokens against support tokens per class

No optimizer step is used. The model performs deterministic inference from support examples only.

## What You Learn

1. How to build support/query prompt structures for classification.
2. How in-context behavior can be prototyped without parameter updates.
3. How to run a CPU-friendly evaluation loop and still log standard lesson artifacts.

## Quick Run

```bash
python -m tracks.nlp.lesson_12_compact_in_context_text_classification.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --num-classes 3 \
  --support-per-class 2 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Expected Outputs

`outputs/nlp/lesson_12_compact_in_context_text_classification/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
