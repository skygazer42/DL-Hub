# Lesson 53: Synthetic Handedness Classification

This toy vision lesson renders compact grayscale hand crops and classifies whether the image
depicts a left or right hand. The synthetic renderer uses a simple palm blob plus a thumb blob
that switches sides, keeping the task CPU friendly and easy to inspect.

Run:

```bash
python -m tracks.vision.lesson_53_synthetic_handedness_classification.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_53_synthetic_handedness_classification/<run_name>/`.

