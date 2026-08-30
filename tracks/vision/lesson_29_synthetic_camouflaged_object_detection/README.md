# Lesson 29: Synthetic Camouflaged-Object Detection

This lesson segments a low-contrast rectangular object embedded in a sinusoidal textured background.
The object reuses the local background patch with only a small intensity delta and boundary cue,
making it a controlled camouflage task generated fully offline.

## Implementation

- `data.py` exposes camouflage contrast and noise controls and returns an image/mask pair.
- `model.py` predicts a dense mask with a compact residual CNN.
- `train.py` optimizes BCE plus Dice loss and measures thresholded mask IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_29_synthetic_camouflaged_object_detection.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_29_synthetic_camouflaged_object_detection/<run_name>/`.
Success requires the standard `config.json`, `metrics.jsonl`, training log, and checkpoint, with
finite BCE/Dice losses and train/eval IoU values in `[0, 1]`.
