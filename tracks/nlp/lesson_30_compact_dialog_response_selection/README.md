# Lesson 30: Compact Dialog Response Selection

This lesson builds a compact response selection setup for task-oriented dialog. Each synthetic
example provides a dialog context and a small set of candidate responses, with exactly one
candidate marked as the best next system reply.

## What It Teaches

- synthetic context-candidate dialog data for retrieval-style response selection
- dual-encoder style scoring with pooled context and candidate representations
- cross-entropy training on candidate logits with top-1 accuracy

## Run

```bash
python -m tracks.nlp.lesson_30_compact_dialog_response_selection.train --device cpu --epochs 1
```
