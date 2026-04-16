# Lesson 42: Face Region Grounding VLM

This lesson builds a toy multimodal grounding model that takes a synthetic face image plus a
region query and predicts the normalized bounding box for the queried face region.

## What It Teaches

- synthetic face-region pairing with deterministic query-conditioned target boxes
- compact vision-language fusion for box regression in normalized image coordinates
- lightweight training loop with box-loss/IoU metrics and checkpoint export
