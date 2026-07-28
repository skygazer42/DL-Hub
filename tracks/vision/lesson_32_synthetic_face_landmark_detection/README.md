# Lesson 32: Synthetic Face Landmark Detection

This compact lesson renders simple grayscale cartoon faces and regresses five normalized landmarks:

- left eye
- right eye
- nose tip
- left mouth corner
- right mouth corner

Run a quick smoke test:

```bash
python -m tracks.vision.lesson_32_synthetic_face_landmark_detection.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/vision/lesson_32_synthetic_face_landmark_detection/<run_name>/`.
