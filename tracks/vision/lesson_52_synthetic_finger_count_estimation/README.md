# Lesson 52: Synthetic Finger Count Estimation

This toy vision lesson renders compact grayscale hand crops and classifies how many fingers are
raised, from `0` through `5`. The synthetic renderer uses a palm blob plus a variable number of
finger blobs so the task stays CPU friendly and easy to inspect.

Run:

```bash
python -m tracks.vision.lesson_52_synthetic_finger_count_estimation.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_52_synthetic_finger_count_estimation/<run_name>/`.
