# Lesson 45: Synthetic Face Identification

This lesson builds a compact face-identification task over small synthetic grayscale face crops.
Each sample is paired with one of five synthetic identities.

## What It Teaches

- deterministic synthetic-data generation for multi-class identity labels
- compact CNN classification for face identification
- basic train/eval loss and accuracy tracking on a tiny vision lesson

## Run

```bash
python -m tracks.vision.lesson_45_synthetic_face_identification.train --device cpu --epochs 1
```
