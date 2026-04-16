# Lesson 25: Toy Low-Shot Intent Detection

This lesson focuses on a small-label regime for intent prediction. The synthetic dataset restricts
each intent to a tiny support budget and asks a compact mean-pooled encoder to generalize from that
budget to paraphrased queries.

## What It Teaches

- intent classification under a low-shot supervision budget
- compact text encoders for small-data adaptation
- lightweight train/eval loops for few-label NLP experiments
