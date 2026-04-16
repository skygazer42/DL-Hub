# Lesson 25: Toy Scene Flow Estimation

This lesson predicts per-point motion between two synthetic point clouds. Each
sample contains a source cloud, a target cloud generated with a deterministic
translation-plus-deformation field, and the supervising scene-flow vectors.

## Run

```bash
python -m tracks.pointcloud.lesson_25_toy_scene_flow_estimation.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_25_toy_scene_flow_estimation --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_25_toy_scene_flow_estimation/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
