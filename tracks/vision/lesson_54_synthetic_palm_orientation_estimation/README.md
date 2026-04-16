# Lesson 54: Synthetic Palm Orientation Estimation

This lesson builds a compact regression task for estimating palm orientation from synthetic
grayscale crops. Each sample renders a palm-like silhouette rotated to a normalized orientation
target in `[0, 1]`, which keeps the training loop small and smoke-test friendly.

The lesson uses a shallow CNN regressor and logs mean-squared loss plus mean absolute error.

## Run

```bash
python -m tracks.vision.lesson_54_synthetic_palm_orientation_estimation.train --device cpu --epochs 1
```

Outputs land under `outputs/vision/lesson_54_synthetic_palm_orientation_estimation/<run_name>/`.
