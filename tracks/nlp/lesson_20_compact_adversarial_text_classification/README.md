# Lesson 20: Compact Adversarial Text Classification

This lesson trains one classifier on paired clean and adversarial sentences. The attack replaces a
class's keyword, context, and action tokens with tokens from the next class while preserving the
original label, creating a controlled robustness test without external attack libraries.

## Implementation

- `data.py` emits clean/adversarial token sequences with a shared synthetic intent label.
- `model.py` applies the same mean-pooled embedding classifier to both views.
- `train.py` combines clean and adversarial cross-entropy with a probability-consistency penalty,
  and reports accuracy for each view.

## Quick Run

```bash
python -m tracks.nlp.lesson_20_compact_adversarial_text_classification.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_20_compact_adversarial_text_classification/<run_name>/`.
A successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite loss and clean/adversarial accuracies in
`[0, 1]` for both training and evaluation.
