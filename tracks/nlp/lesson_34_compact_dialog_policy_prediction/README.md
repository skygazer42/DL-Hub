# Lesson 34: Compact Dialog Policy Prediction

This lesson implements a compact policy classifier for short task-oriented dialog context.
Each synthetic example represents a user and system exchange with a next policy action label.

The lesson covers:
- deterministic synthetic dialog policy data with five policy classes
- pooled-embedding text classification
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_34_compact_dialog_policy_prediction.train --device cpu --epochs 1
```
