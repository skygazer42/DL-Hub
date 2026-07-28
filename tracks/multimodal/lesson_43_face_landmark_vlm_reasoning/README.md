# Lesson 43: Face Landmark VLM Reasoning

This lesson builds a compact multimodal reasoner that takes a synthetic face image plus a
landmark query and predicts the normalized `(x, y)` location for the queried face landmark.

## What It Teaches

- deterministic synthetic face generation with landmark-coordinate supervision
- compact vision-language fusion for query-conditioned 2D coordinate regression
- lightweight training loop with point-distance metrics and checkpoint export
