# Lesson 44: Toy Dialog SLA Breach Prediction

This lesson classifies whether a support interaction is likely to breach its SLA bucket. The
synthetic text keeps the signal explicit so the full pipeline remains CPU friendly and easy to
inspect in tests.

Each example is tokenized into a fixed-length sequence and labeled as either `ok` or `breach`.
Tokens such as `sla`, `breach`, and `minutes` are baked into the corpus so the batch contract is
easy to inspect in tests.

Run:

```bash
python -m tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.train --device cpu --epochs 1
```

Outputs are written to
`outputs/nlp/lesson_44_toy_dialog_sla_breach_prediction/<run_name>/`.
