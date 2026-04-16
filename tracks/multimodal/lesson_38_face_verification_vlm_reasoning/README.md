# Lesson 38: Face Verification VLM Reasoning

This lesson builds a toy multimodal verifier that compares two synthetic face crops and a short
query prompt, then predicts whether both faces belong to the same identity.

## What It Teaches

- synthetic pair generation for same-identity vs different-identity verification
- compact vision-language fusion over paired face evidence and textual query
- binary decision training with lightweight metrics and checkpointing
