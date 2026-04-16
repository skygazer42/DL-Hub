# Lesson 28: Toy 3D Semantic Segmentation

This lesson performs per-point semantic segmentation on tiny synthetic point
clouds. Samples are generated from a deterministic sphere-or-cube surface
distribution, and each point receives a semantic label from angle-based sectors.

## Run

```bash
python -m tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_28_toy_3d_semantic_segmentation --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_28_toy_3d_semantic_segmentation/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
