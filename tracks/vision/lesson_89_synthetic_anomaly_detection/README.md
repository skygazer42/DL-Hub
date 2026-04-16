# Lesson 89: Synthetic Vision Anomaly Detection

This lesson adds a toy-first, CPU-friendly anomaly detection pipeline on synthetic images.

## What it covers

- synthetic normal/anomalous image generation with pixel anomaly maps
- tiny anomaly detector from `dlhub.vision.anomaly_detection.patchcore`
- joint supervision on reconstruction, anomaly map, and image-level anomaly score

## Run

```bash
python -m tracks.vision.lesson_89_synthetic_anomaly_detection.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

