# Lesson 31: Toy Open-Vocabulary 3D Recognition and Grounding

This lesson builds a compact text-conditioned point-cloud model over synthetic
two-object scenes. Given a query such as "locate the crimson cube target", the
model predicts:

- the queried object category (`cube`, `sphere`, or `cylinder`)
- a per-point grounding mask for the target object

The toy setup uses synonym variants in the text prompts to mimic open-vocabulary
behavior while staying lightweight for CPU smoke runs.

## Run

```bash
python -m tracks.pointcloud.lesson_31_toy_open_vocabulary_3d.train --run-name dev
python scripts/run_lesson.py pointcloud lesson_31_toy_open_vocabulary_3d --dry-run
```

## Outputs

Runs are written to `outputs/pointcloud/lesson_31_toy_open_vocabulary_3d/<run-name>/`
with `config.json`, `metrics.jsonl`, logs, and a checkpoint.
