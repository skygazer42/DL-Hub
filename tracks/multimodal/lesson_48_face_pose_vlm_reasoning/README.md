# Lesson 48: Face Pose VLM Reasoning

This lesson builds a compact multimodal face pose estimator that consumes a synthetic face crop plus
an instruction-like query and predicts normalized yaw, pitch, and roll.

## What It Teaches

- deterministic synthetic face generation with pose-conditioned appearance cues
- lightweight vision-language fusion for query-conditioned face pose regression
- CPU-friendly training with SmoothL1 and mean absolute pose error tracking
