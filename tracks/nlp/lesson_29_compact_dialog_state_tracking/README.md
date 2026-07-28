# Lesson 29: Compact Dialog State Tracking

This lesson introduces a compact dialog state tracking setup over short task-oriented restaurant
dialogs. Each example contains a few turns of user and system text, and the target state tracks
the final `cuisine`, `area`, and `party_size` slot values after possible user corrections.

## What It Teaches

- synthetic multi-turn dialog generation with slot-value updates
- pooled text encoding with one classification head per tracked slot
- slot-level and joint-goal accuracy for simple dialog state tracking

## Run

```bash
python -m tracks.nlp.lesson_29_compact_dialog_state_tracking.train --device cpu --epochs 1
```
