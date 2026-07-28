# Lesson 41: Face Occlusion VLM Reasoning

This lesson builds a compact multimodal reasoner that takes a synthetic face crop plus a short
occlusion query and predicts whether the visible face is lightly or heavily occluded.

## What It Teaches

- synthetic face generation with deterministic occluders and occlusion-ratio targets
- compact vision-language fusion for query-conditioned occlusion reasoning
- lightweight binary training loop with metrics logging and checkpoint export
