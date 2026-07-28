# Lesson 33: Synthetic Face Liveness Detection

This compact lesson renders grayscale cartoon faces and classifies whether each sample looks like a
live capture or a spoofed presentation attack.

Spoof examples inject cues such as:

- display borders
- stripe artifacts
- reduced local contrast

Run a quick smoke test:

```bash
python -m tracks.vision.lesson_33_synthetic_face_liveness_detection.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/vision/lesson_33_synthetic_face_liveness_detection/<run_name>/`.
