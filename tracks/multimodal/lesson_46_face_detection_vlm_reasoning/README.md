# Lesson 46: Face Detection VLM Reasoning

This lesson builds a compact multimodal detector that takes a synthetic face image plus a short
query and predicts a normalized face bounding box.

## What It Teaches

- synthetic face image generation with normalized XYXY detection targets
- compact query-conditioned detector for face box prediction
- lightweight SmoothL1+IoU training loop with checkpoint export
