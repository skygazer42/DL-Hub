# Lesson 35: Compact Dialog Domain Prediction

This lesson implements a compact domain classifier for task-oriented dialog. Each synthetic
example is a short dialog context paired with one of four domains:
`restaurant`, `taxi`, `hotel`, or `weather`.

The lesson covers:
- deterministic synthetic dialog data with four domain labels
- pooled-embedding text classification
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_35_compact_dialog_domain_prediction.train --device cpu --epochs 1
```
