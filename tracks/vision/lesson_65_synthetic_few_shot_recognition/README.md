# Lesson 65: Synthetic Few-Shot Recognition

This lesson builds a compact prototypical few-shot recognizer on deterministic
synthetic grayscale shape episodes. Each episode samples a subset of shape
classes, renders a small support set plus query set, and trains a tiny CNN
encoder with prototype classification.

## Run

```bash
python -m tracks.vision.lesson_65_synthetic_few_shot_recognition.train \
  --device cpu \
  --epochs 3 \
  --num-episodes 256 \
  --num-ways 4 \
  --shots 2 \
  --queries-per-class 3
```

Or resolve the lesson entrypoint through the shared helper:

```bash
python scripts/run_lesson.py vision lesson_65_synthetic_few_shot_recognition --dry-run
```

## Outputs

- `outputs/vision/lesson_65_synthetic_few_shot_recognition/<run_name>/config.json`
- `outputs/vision/lesson_65_synthetic_few_shot_recognition/<run_name>/metrics.jsonl`
- `outputs/vision/lesson_65_synthetic_few_shot_recognition/<run_name>/logs/train.log`
- `outputs/vision/lesson_65_synthetic_few_shot_recognition/<run_name>/checkpoints/checkpoint.pt`
