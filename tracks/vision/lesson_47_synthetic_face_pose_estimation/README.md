# Lesson 47: Synthetic Face Pose Estimation

This compact lesson renders simple grayscale cartoon faces and regresses a compact three-value head
pose vector:

- yaw
- pitch
- roll

Each target is normalized to `[-1, 1]` so the lesson stays small and CPU-friendly while still
teaching pose-style dense regression.

Run:

```bash
python -m tracks.vision.lesson_47_synthetic_face_pose_estimation.train \
  --device cpu --epochs 1
```

Outputs land under `outputs/vision/lesson_47_synthetic_face_pose_estimation/<run_name>/`.
