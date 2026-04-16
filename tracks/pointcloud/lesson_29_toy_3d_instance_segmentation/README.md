# Lesson 29: Toy 3D Instance Segmentation

This lesson trains a tiny per-point network on synthetic point clouds composed
of two Gaussian clusters. The task is to assign each point to one of two
instance ids.

## Run

```bash
python -m tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_29_toy_3d_instance_segmentation --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_29_toy_3d_instance_segmentation/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
