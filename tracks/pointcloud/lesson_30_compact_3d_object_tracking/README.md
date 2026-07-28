# Lesson 30: Compact 3D Object Tracking

This lesson tracks a synthetic object across two pointcloud frames. The model
predicts a compact 6D state made of current object center `(x, y, z)` and
velocity `(vx, vy, vz)`.

## Run

```bash
python -m tracks.pointcloud.lesson_30_compact_3d_object_tracking.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_30_compact_3d_object_tracking --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_30_compact_3d_object_tracking/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
