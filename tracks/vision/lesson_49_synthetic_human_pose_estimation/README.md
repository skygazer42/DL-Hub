# Lesson 49: Synthetic Human Pose Estimation

This toy lesson renders simple grayscale stick figures and regresses eight normalized human
keypoints:

- head
- left shoulder
- right shoulder
- left hand
- right hand
- pelvis
- left foot
- right foot

It follows the same small, CPU-friendly regression pattern as the nearby synthetic face landmark
and face pose lessons.

Run a quick smoke test:

```bash
python -m tracks.vision.lesson_49_synthetic_human_pose_estimation.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/vision/lesson_49_synthetic_human_pose_estimation/<run_name>/`.
