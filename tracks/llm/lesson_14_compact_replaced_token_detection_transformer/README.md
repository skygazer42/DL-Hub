# Lesson 14: Compact Replaced-Token Detection Transformer

This compact lesson demonstrates a small self-supervised pretraining objective:

- Corrupt synthetic sequences by replacing a subset of content tokens.
- Train a tiny transformer with two heads:
  - token reconstruction head (predict clean token ids)
  - replaced-token detection head (predict whether each token was replaced)

The setup is CPU-friendly and intended for quick smoke runs.
