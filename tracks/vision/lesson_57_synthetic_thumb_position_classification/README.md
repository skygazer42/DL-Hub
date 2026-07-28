# Lesson 57: Synthetic Thumb Position Classification

This compact vision lesson renders compact grayscale hand crops and classifies a coarse thumb
position state. The synthetic renderer keeps the hand structure simple: a palm blob, four
finger-like blobs, and a thumb blob that moves across three discrete vertical positions.

Run:

```bash
python -m tracks.vision.lesson_57_synthetic_thumb_position_classification.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_57_synthetic_thumb_position_classification/<run_name>/`.
