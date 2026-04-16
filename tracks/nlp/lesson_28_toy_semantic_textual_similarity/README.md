# Lesson 28: Toy Semantic Textual Similarity

This lesson builds a compact semantic textual similarity (STS) regressor on synthetic
sentence-pair examples. It predicts a normalized similarity score in `[0, 1]` from lightweight
token features and pooled embeddings.

## What It Teaches

- sentence-pair regression for semantic similarity
- compact shared text encoder + scalar regression head
- minimal NLP training loop with MSE optimization and MAE reporting
