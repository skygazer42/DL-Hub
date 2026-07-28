# Lesson 32: Compact Dialog Act Prediction

This lesson introduces a compact dialog act classifier for short task-oriented turns. Each
synthetic example contains a small speaker-tagged dialog snippet and a label for the next-turn act
such as greeting, request, or confirmation.

## What It Teaches

- synthetic dialog-turn generation with diverse intent-like act labels
- pooled token embedding encoder with a multiclass classification head
- cross-entropy training with top-1 accuracy for dialog act prediction

## Run

```bash
python -m tracks.nlp.lesson_32_compact_dialog_act_prediction.train --device cpu --epochs 1
```
