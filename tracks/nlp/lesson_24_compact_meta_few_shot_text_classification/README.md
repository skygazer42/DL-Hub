# Lesson 24: Compact Meta Few-Shot Text Classification

This lesson turns the earlier episodic few-shot block into an explicitly meta-learning flavored
exercise. Each synthetic episode exposes a small support set and a query set for a handful of
intent-style tasks, and the model learns prototype-based adaptation from those episodes.

## What It Teaches

- episodic support/query construction for text tasks
- prototype formation from support embeddings
- query classification through metric-space adaptation
- compact evaluation loops for meta few-shot NLP experiments
