# Lesson 80: Synthetic Event Camera Understanding

This lesson builds a tiny CPU-friendly event understanding pipeline:

- synthetic event volume generation with moving blobs
- multitask supervision: polarity map, motion field, and depth-like confidence
- small model that uses a `dlhub.vision.event_camera_understanding` backbone family

Run:

```bash
python -m tracks.vision.lesson_80_synthetic_event_camera_understanding.train --epochs 1 --max-train-batches 2 --max-eval-batches 1
```
