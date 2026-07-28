# Lesson 40: Compact Dialog Priority Prediction

This lesson implements a compact classifier that predicts `low`, `medium`, or `high`
priority from a short task-oriented support conversation.

The synthetic dataset uses repetitive support-dialog phrases so the lesson stays fast on CPU and
easy to inspect. Key tokens such as `support`, `dialog`, `priority`, and `urgent` are intentionally
included to keep the compact vocabulary interpretable.

## Run

```bash
python -m tracks.nlp.lesson_40_compact_dialog_priority_prediction.train --device cpu --epochs 1
```

Outputs land under `outputs/nlp/lesson_40_compact_dialog_priority_prediction/<run_name>/`.
