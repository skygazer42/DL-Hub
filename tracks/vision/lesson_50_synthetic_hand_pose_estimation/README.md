# Lesson 50: Synthetic Hand Pose Estimation

This toy lesson renders a compact grayscale hand skeleton and regresses ten normalized keypoints:
one wrist anchor, a thumb tip, and four finger base-tip pairs. The renderer is deterministic given
the seed and stays light enough for CPU-only smoke tests.

The training loop follows the same regression shell used by neighboring vision lessons and writes
the standard artifacts expected by the repo's lesson runner.

Run:

```bash
python -m tracks.vision.lesson_50_synthetic_hand_pose_estimation.train --epochs 1 --device cpu
```

Outputs land under `outputs/vision/lesson_50_synthetic_hand_pose_estimation/<run_name>/`.
