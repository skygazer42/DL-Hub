# Lesson 59: Synthetic Thumb Contact Classification

This compact vision lesson renders compact grayscale hand crops and classifies whether the thumb
is touching the palm. The synthetic renderer keeps the hand structure simple: a palm blob,
four finger-like blobs, and a thumb blob that either stays separated from the palm or forms
a soft contact bridge into it.

Run:

```bash
python -m tracks.vision.lesson_59_synthetic_thumb_contact_classification.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_59_synthetic_thumb_contact_classification/<run_name>/`.
