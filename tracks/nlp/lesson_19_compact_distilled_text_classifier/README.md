# Lesson 19: Compact Distilled Text Classifier

This lesson compares a small student embedding classifier with a wider teacher branch on four
synthetic intent classes. Both branches are trained jointly from scratch: the teacher contributes
supervised cross-entropy while its detached soft distribution guides the student. It therefore
demonstrates online distillation mechanics rather than compression from a pretrained teacher.

## Implementation

- `data.py` generates class-specific keyword/context/action sentences and a local vocabulary.
- `model.py` provides separate student and teacher embedding/head paths.
- `train.py` combines student/teacher cross-entropy with temperature-scaled KL distillation loss.

## Quick Run

```bash
python -m tracks.nlp.lesson_19_compact_distilled_text_classifier.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_19_compact_distilled_text_classifier/<run_name>/`. A
successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite `train_distill_loss`, `train_ce_loss`, and
evaluation loss, with student accuracies in `[0, 1]`.
