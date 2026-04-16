# Lesson 27: Toy Textual Entailment

This lesson builds a tiny textual entailment classifier over synthetic premise-hypothesis pairs.
It uses a compact mean-pooled encoder with a 3-way label space: entailment, contradiction, and
neutral.

## What It Teaches

- premise-hypothesis pair classification with lightweight text features
- compact encoder + classifier design for CPU-friendly experiments
- minimal train/eval loop and artifact logging conventions in the NLP track
