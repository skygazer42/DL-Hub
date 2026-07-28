# Lesson 26: Compact Joint Intent + Slot Parsing

This lesson builds a compact task-oriented NLU pipeline that predicts utterance-level intent and
token-level BIO slots in one model. The synthetic data covers common flight-assistant requests and
uses slot spans for `from_city`, `to_city`, and `date`.

## What It Teaches

- multitask learning for joint intent classification and slot tagging
- BIO span labeling with simple token-level supervision
- lightweight, CPU-friendly train/eval loops for structured NLU
