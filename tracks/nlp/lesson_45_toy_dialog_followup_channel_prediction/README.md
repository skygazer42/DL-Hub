# Lesson 45: Toy Dialog Followup Channel Prediction

This lesson trains a compact text classifier that predicts the best followup channel for a support
dialog. Each synthetic example encodes issue type, customer mood, current status, and a preferred
followup route, then maps the exchange to one of three labels: `email`, `sms`, or `call`.

The lesson is intentionally small and deterministic so it can act as a fast smoke-testable teaching
example for multiclass dialog routing workflows.

## Run

```bash
python -m tracks.nlp.lesson_45_toy_dialog_followup_channel_prediction.train --device cpu --epochs 1
```

Outputs land under
`outputs/nlp/lesson_45_toy_dialog_followup_channel_prediction/<run_name>/`.
