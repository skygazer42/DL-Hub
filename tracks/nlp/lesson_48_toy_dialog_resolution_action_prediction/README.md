# Lesson 48: Toy Dialog Resolution Action Prediction

This lesson predicts the next resolution action for a short synthetic dialog summary. The model pools
token embeddings and classifies among `close`, `handoff`, `followup`, `resolve`, and `escalate`.

## Run

```bash
python -m tracks.nlp.lesson_48_toy_dialog_resolution_action_prediction.train --epochs 1 --device cpu
```

Outputs are written to `outputs/nlp/lesson_48_toy_dialog_resolution_action_prediction/<run_name>/`.
