# Lesson 22: Compact Weak-Supervision Text Classification

This lesson keeps the text-classification setup small, but replaces fully supervised labels with
noisy labeling-function votes. The model fuses pooled text features with the vote pattern and
learns against soft pseudo-label probabilities.

Quick smoke run:

```bash
python -m tracks.nlp.lesson_22_compact_weak_supervision_text_classification.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/nlp/lesson_22_compact_weak_supervision_text_classification/<run_name>/`.
