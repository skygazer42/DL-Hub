# Lesson 43: Compact Dialog Callback Prediction

This lesson classifies whether a support interaction will require a callback follow-up. The
synthetic text keeps the signal explicit so the full pipeline remains CPU friendly and easy to
inspect in tests.

Each example is tokenized into a fixed-length sequence and labeled as either `no_callback` or
`callback`. Tokens such as `customer`, `callback`, and `followup` are baked into the corpus so the
batch contract is easy to inspect in tests.

Run:

```bash
python -m tracks.nlp.lesson_43_compact_dialog_callback_prediction.train --device cpu --epochs 1
```

Outputs are written to
`outputs/nlp/lesson_43_compact_dialog_callback_prediction/<run_name>/`.
