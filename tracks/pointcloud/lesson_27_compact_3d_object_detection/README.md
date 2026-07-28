# Lesson 27: Compact 3D Object Detection

This lesson trains a compact pointcloud detector on synthetic 3D scenes. Each
sample contains one object point cluster (cuboid or ellipsoid), uniform clutter
points, a class label, and a 7D bounding box target `(cx, cy, cz, dx, dy, dz, yaw)`.

## Run

```bash
python -m tracks.pointcloud.lesson_27_compact_3d_object_detection.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_27_compact_3d_object_detection --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_27_compact_3d_object_detection/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
