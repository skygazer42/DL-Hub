# Lesson 44: Face Parsing VLM Reasoning

This lesson builds a compact multimodal parser that takes a synthetic face image plus a short
query and predicts a binary mask for the queried face part (eyes, mouth, hair, or skin).

## What It Teaches

- synthetic face-part mask generation with query-conditioned mask targets
- compact text-conditioned segmentation head for coarse face parsing
- lightweight BCE+Dice training loop with IoU metric logging and checkpoint export
