# Lesson 35: Face Expression VLM Recognition

This lesson pairs compact synthetic face features with a short text prompt and trains a tiny
multimodal classifier to recognize one of four facial expressions: `happy`, `sad`, `angry`, or
`neutral`.

## What It Teaches

- prompt-conditioned emotion recognition with lightweight multimodal fusion
- toy feature synthesis for expression-aware face embeddings
- CPU-friendly train/eval loops with reproducible outputs

## Run

```bash
python -m tracks.multimodal.lesson_35_face_expression_vlm_recognition.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```
