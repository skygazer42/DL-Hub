# Lesson 26: Compact Gaussian Splatting

This lesson trains a tiny point-cloud encoder that predicts per-point 2D Gaussian
splat parameters and renders a compact density image target.

## Run

```bash
python -m tracks.pointcloud.lesson_26_compact_gaussian_splatting.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_26_compact_gaussian_splatting --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_26_compact_gaussian_splatting/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
