# Lesson 73: Synthetic Video Object Segmentation

This lesson segments one bright rectangle as it moves through a noisy grayscale clip. The object
size, velocity, and every per-frame binary mask are generated from the sample seed.

## Implementation

- `data.py` returns clips and dense masks shaped by time.
- `model.py` predicts one foreground logit map for every frame with a residual video segmenter.
- `train.py` combines BCE and Dice losses and reports spatiotemporal mask IoU.

## Quick Run

```bash
python -m tracks.vision.lesson_73_synthetic_video_object_segmentation.train \
  --epochs 1 --num-samples 48 --batch-size 4 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_73_synthetic_video_object_segmentation/<run_name>/`. Success
requires the standard config, metrics, log, and checkpoint artifacts, finite BCE/Dice losses, and
train/eval IoU values in `[0, 1]`.
