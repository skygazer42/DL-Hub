# Lesson 46: Compact Dialog Reopen Prediction

This lesson trains a compact text classifier that predicts whether a support dialog will reopen
after an apparent resolution. The synthetic samples encode issue type, channel, mood, and closure
signals, then map each exchange to one of two labels: `closed` or `reopen`.

The lesson is intentionally small and deterministic so it can serve as a fast, smoke-testable
example for pooled-embedding dialog status classification workflows.

## Run

```bash
python -m tracks.nlp.lesson_46_compact_dialog_reopen_prediction.train --device cpu --epochs 1
```

Outputs land under `outputs/nlp/lesson_46_compact_dialog_reopen_prediction/<run_name>/`.
