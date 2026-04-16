# Lesson 47: Toy Dialog Resolution Owner Prediction

This lesson predicts which owner should resolve a short synthetic dialog summary. The model pools
token embeddings and classifies among `billing`, `support`, and `operations`.

## Run

```bash
python -m tracks.nlp.lesson_47_toy_dialog_resolution_owner_prediction.train --epochs 1 --device cpu
```

Outputs are written to `outputs/nlp/lesson_47_toy_dialog_resolution_owner_prediction/<run_name>/`.
