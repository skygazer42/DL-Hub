# Lesson 56: Synthetic Finger Spread Estimation

This lesson regresses the normalized spread of a synthetic hand silhouette from a grayscale crop.
The dataset returns `(image, target)` pairs with a scalar target in `[0, 1]`.

## Run

```bash
python -m tracks.vision.lesson_56_synthetic_finger_spread_estimation.train --epochs 1 --device cpu
```

Outputs are written to `outputs/vision/lesson_56_synthetic_finger_spread_estimation/<run_name>/`.
