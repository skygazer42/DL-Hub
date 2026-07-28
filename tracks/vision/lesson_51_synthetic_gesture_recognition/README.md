# Lesson 51: Synthetic Gesture Recognition

This compact lesson renders simple grayscale stick figures and classifies four deterministic gesture
states from arm and hand geometry:

- `rest`
- `left_wave`
- `right_wave`
- `hands_up`

The dataset, model, and training loop are intentionally small and CPU-friendly, following the same
classification pattern as the nearby synthetic face-expression lesson.

Run a quick smoke test:

```bash
python -m tracks.vision.lesson_51_synthetic_gesture_recognition.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/vision/lesson_51_synthetic_gesture_recognition/<run_name>/`.
