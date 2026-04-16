# Lesson 58: Synthetic Finger Curvature Estimation

This lesson regresses a normalized finger-curvature score from a compact grayscale hand crop.
The synthetic generator bends fingertip blobs more aggressively as curvature increases.

## Run

```bash
python -m tracks.vision.lesson_58_synthetic_finger_curvature_estimation.train --epochs 1 --device cpu
```

Outputs are written to `outputs/vision/lesson_58_synthetic_finger_curvature_estimation/<run_name>/`.
