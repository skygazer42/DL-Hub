# Lesson 31: Toy Slot Carryover Prediction

This lesson introduces a compact slot carryover prediction setup over short task-oriented
restaurant dialogs. Each example contains a small dialog history plus a follow-up utterance, and
the targets indicate whether `cuisine`, `area`, and `party` slots should be carried from history
into the current turn.

## What It Teaches

- synthetic history-plus-followup dialog generation for carryover decisions
- pooled text encoding with one binary classification head per slot
- slot-level and joint carryover accuracy for simple dialog understanding

## Run

```bash
python -m tracks.nlp.lesson_31_toy_slot_carryover_prediction.train --device cpu --epochs 1
```
