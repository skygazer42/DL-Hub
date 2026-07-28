# Lesson 17 - Synthetic Super-Resolution

This lesson trains a compact-first paired super-resolution model on synthetic images.

It is designed to be CPU-friendly and self-contained. No external dataset download is required.

## Run

```bash
python -m tracks.vision.lesson_17_synthetic_super_resolution.train \
  --arch sr:srcnn_tiny \
  --epochs 1 \
  --image-size 32 \
  --batch-size 4 \
  --device cpu \
  --run-name dev
```

Recommended starter arches:

- `sr:srcnn_tiny`
- `sr:edsr_sr_tiny`

Outputs land in:

`outputs/vision/lesson_17_synthetic_super_resolution/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `checkpoints/checkpoint.pt`
- `predictions.pt`
- `preview.png` (optional, if `torchvision` is installed)
