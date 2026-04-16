# Lesson 47: Face Retrieval VLM Reasoning

This lesson trains a tiny multimodal retriever that combines a synthetic face crop with a short
query and predicts which identity should be retrieved from a small gallery.

## What It Teaches

- deterministic synthetic face generation with identity-specific facial cues
- compact vision-language fusion for gallery-style face retrieval
- CPU-friendly training with cross-entropy loss and top-1 retrieval metrics
