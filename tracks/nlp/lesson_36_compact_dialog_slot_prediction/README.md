# Lesson 36: Compact Dialog Slot Prediction

This lesson implements a compact multi-head slot predictor for task-oriented dialog.
Each synthetic example is a short dialog context with three slot targets:
`cuisine`, `area`, and `party`.

The lesson covers:
- deterministic synthetic slot-labeled dialog data
- pooled-embedding text encoder with per-slot classification heads
- config, vocab, metrics, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.nlp.lesson_36_compact_dialog_slot_prediction.train --device cpu --epochs 1
```
