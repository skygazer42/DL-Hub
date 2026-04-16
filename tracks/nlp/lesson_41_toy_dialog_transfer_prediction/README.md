# Lesson 41: Toy Dialog Transfer Prediction

This lesson classifies whether a short support dialog should stay with the current agent, receive
light specialist review, or be transferred immediately. The synthetic dataset keeps the signal
compact and explicit so the training loop remains CPU friendly.

Each example is tokenized into a fixed-length sequence and labeled with one of three transfer
levels: `low`, `medium`, or `high`. Tokens such as `transfer`, `specialist`, and `agent` are baked
into the corpus so the batch contract is easy to inspect in tests.

Run:

```bash
python -m tracks.nlp.lesson_41_toy_dialog_transfer_prediction.train --device cpu --epochs 1
```

Outputs are written to
`outputs/nlp/lesson_41_toy_dialog_transfer_prediction/<run_name>/`.
