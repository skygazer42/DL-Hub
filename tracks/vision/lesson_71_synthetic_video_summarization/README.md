# Lesson 71: Synthetic Video Summarization

This lesson assigns an importance score to each frame of a short clip. Random key frames contain an
additional bright square while all frames share a moving blob, giving an explicit frame-selection
target for a fully offline summarization task.

## Implementation

- `data.py` returns a clip and binary `importance` vector with configurable key-frame count.
- `model.py` encodes each frame and predicts one importance logit per timestep.
- `train.py` reports importance loss and frame-score mean absolute error.

## Quick Run

```bash
python -m tracks.vision.lesson_71_synthetic_video_summarization.train \
  --epochs 1 --num-samples 64 --batch-size 8 --num-key-frames 2 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_71_synthetic_video_summarization/<run_name>/`. Success requires
the standard config, metrics, log, and checkpoint artifacts, finite importance loss, and
non-negative train/eval MAE.
