# Lesson 45: Face Alignment VLM Reasoning

This lesson builds a compact multimodal aligner that takes a synthetic face image plus a short query
and predicts a canonical five-point facial landmark layout.

## What It Teaches

- deterministic synthetic face generation with pose-jittered five-point landmarks
- compact vision-language fusion for query-conditioned landmark alignment
- lightweight regression training loop with mean landmark L2 metrics and checkpoint export
