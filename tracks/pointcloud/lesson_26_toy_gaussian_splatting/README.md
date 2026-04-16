# Lesson 26: Toy Gaussian Splatting

This lesson trains a tiny point-cloud encoder that predicts per-point 2D Gaussian
splat parameters and renders a toy density image target.

## Run

```bash
python -m tracks.pointcloud.lesson_26_toy_gaussian_splatting.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_26_toy_gaussian_splatting --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_26_toy_gaussian_splatting/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
