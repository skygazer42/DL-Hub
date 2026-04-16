# Lesson 48: Synthetic Gaze Estimation

This lesson builds a compact gaze-regression task from deterministic grayscale face crops. Each
sample contains a rendered face with eye pupils shifted by a normalized `(x, y)` gaze target in the
range `[0, 1]`.

The training loop uses a small CNN regressor with an MSE objective and reports both regression loss
and average L1 error.

## Run

```bash
python -m tracks.vision.lesson_48_synthetic_gaze_estimation.train --device cpu --epochs 1
```

Outputs land under `outputs/vision/lesson_48_synthetic_gaze_estimation/<run_name>/`.
