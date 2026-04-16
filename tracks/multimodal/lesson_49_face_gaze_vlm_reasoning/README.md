# Lesson 49: Face Gaze VLM Reasoning

This lesson builds a toy multimodal regressor that combines synthetic face evidence with a short
query/context prompt to infer a compact normalized gaze target on a shared screen plane.

## What It Teaches

- deterministic synthetic face generation with eye-direction cues and face-box side information
- lightweight vision-language fusion for context-conditioned gaze target regression
- CPU-friendly training with smooth regression loss, mean absolute error tracking, and small output artifacts
