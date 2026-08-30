# Lesson 25: Synthetic Multi-Focus Image Fusion

This lesson reconstructs an all-in-focus RGB image from complementary near-focus and far-focus
views. The offline renderer assigns depths to colored circles and rectangles, then blurs opposite
depth regions to create the paired inputs.

## Implementation

- `data.py` returns `near_focus`/`far_focus` tensors and the original clean target.
- `model.py` concatenates both views and applies a compact residual CNN with a sigmoid RGB head.
- `train.py` combines L1 reconstruction with spatial-gradient consistency and reports PSNR.

## Quick Run

```bash
python -m tracks.vision.lesson_25_synthetic_image_fusion.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

Runs write to `outputs/vision/lesson_25_synthetic_image_fusion/<run_name>/`. Success requires exit
code 0 plus `config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`.
Metrics must contain finite reconstruction/consistency losses and train/eval PSNR values.
