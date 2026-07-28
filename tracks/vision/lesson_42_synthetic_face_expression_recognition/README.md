# Lesson 42: Synthetic Face Expression Recognition

This lesson builds a compact expression-recognition task over small synthetic grayscale face crops.
Each sample is paired with a single expression class:
`0=neutral`, `1=happy`, `2=sad`, `3=surprised`.

## What It Teaches

- deterministic synthetic-data generation for multi-class facial expression labels
- compact CNN classification for face expression recognition
- basic train/eval loss and accuracy tracking on a tiny vision lesson

## Run

```bash
python -m tracks.vision.lesson_42_synthetic_face_expression_recognition.train --device cpu --epochs 1
```
