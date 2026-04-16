# Lesson 55: Synthetic Sign Digit Classification

This toy vision lesson renders compact grayscale hand crops and classifies a synthetic sign-digit
label (`0` through `9`). The renderer intentionally stays simple: a palm blob plus a few
finger-like blobs, and a digit-specific corner marker so the 10-way task remains CPU friendly and
stable for smoke tests.

Run:

```bash
python -m tracks.vision.lesson_55_synthetic_sign_digit_classification.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_55_synthetic_sign_digit_classification/<run_name>/`.
