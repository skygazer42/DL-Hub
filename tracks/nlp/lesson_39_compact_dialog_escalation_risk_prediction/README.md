# Lesson 39: Compact Dialog Escalation Risk Prediction

This lesson implements a compact dialog-state classifier that predicts whether a support
conversation has `low`, `medium`, or `high` escalation risk.

The synthetic texts are short and intentionally repetitive so the lesson stays CPU-friendly and easy
to inspect. Key tokens such as `dialog`, `agent`, `escalation`, `risk`, and `urgent` are baked into
the compact corpus to make the vocabulary and labels interpretable.

## Run

```bash
python -m tracks.nlp.lesson_39_compact_dialog_escalation_risk_prediction.train --device cpu --epochs 1
```

Outputs land under
`outputs/nlp/lesson_39_compact_dialog_escalation_risk_prediction/<run_name>/`.
