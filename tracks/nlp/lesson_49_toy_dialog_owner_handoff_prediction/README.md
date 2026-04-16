# Lesson 49: Toy Dialog Owner Handoff Prediction

This lesson predicts whether a short dialog should stay in the current queue or hand off to
`billing`, `support`, or `operations` before closure.

## Run

```bash
python -m tracks.nlp.lesson_49_toy_dialog_owner_handoff_prediction.train --epochs 1 --device cpu
```

Outputs are written to `outputs/nlp/lesson_49_toy_dialog_owner_handoff_prediction/<run_name>/`.
