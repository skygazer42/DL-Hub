# Lesson 40: Synthetic Face Attribute Recognition

This lesson frames face-attribute recognition as multi-label classification over small synthetic
face crops. Each sample pairs a rendered grayscale face with three binary attributes:
`[smiling, glasses, beard]`.

## What It Teaches

- generating deterministic synthetic multi-label attribute targets
- training a compact CNN for multi-label face attributes
- tracking validation attribute exact-match accuracy during training

## Run

```bash
python -m tracks.vision.lesson_40_synthetic_face_attribute_recognition.train --device cpu --epochs 1
```
