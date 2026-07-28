# Lesson 38: Compact Dialog Satisfaction Prediction

This lesson implements a compact dialog satisfaction classifier for task-oriented support
dialogs. Each synthetic example is a short dialog summary paired with one of three satisfaction
levels: `dissatisfied`, `neutral`, or `satisfied`.

The lesson covers:
- deterministic synthetic dialog data with three satisfaction labels
- pooled-embedding text classification
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_38_compact_dialog_satisfaction_prediction.train --device cpu --epochs 1
```
