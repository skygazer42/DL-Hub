# Lesson 37: Toy Dialog Outcome Prediction

This lesson implements a compact outcome classifier for task-oriented dialog. Each synthetic
example is a short dialog context paired with one of three outcomes:
`resolved`, `pending`, or `escalated`.

The lesson covers:
- deterministic synthetic dialog data with three outcome labels
- pooled-embedding text classification
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_37_toy_dialog_outcome_prediction.train --device cpu --epochs 1
```
