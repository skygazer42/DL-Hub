# Lesson 43: Synthetic Deepfake Detection

This lesson builds a compact binary image classifier for synthetic deepfake detection. Each
sample is a toy face image that is either rendered as a clean capture or perturbed with
artifacts that mimic blending, over-smoothing, and generative texture errors.

The lesson covers:
- deterministic synthetic face generation for binary classification
- a lightweight CNN classifier with cross-entropy training
- config, metrics, logs, and checkpoint outputs

Run locally with:

```bash
python -m tracks.vision.lesson_43_synthetic_deepfake_detection.train --device cpu --epochs 1
```
