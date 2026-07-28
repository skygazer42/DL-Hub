# Lesson 33: Compact Dialog Intent Prediction

This lesson implements a compact intent classifier for task-oriented dialog. Each synthetic
example is a short dialog context paired with one of four intents, such as restaurant booking
or taxi cancellation.

The lesson covers:
- deterministic synthetic dialog data with four intent labels
- pooled-embedding text classification
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_33_compact_dialog_intent_prediction.train --device cpu --epochs 1
```
