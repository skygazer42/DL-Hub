# Lesson 11: Toy Few-Shot Text Classification

This lesson introduces episodic few-shot learning for NLP with a small prototypical-network style
classifier.

## What You Learn

- how to build support/query episodes instead of ordinary iid batches
- how a shared text encoder can produce per-class prototypes
- how query classification becomes distance-to-prototype matching

## Files

- `data.py`: synthetic intent episodes with `num_ways`, `shots`, and `queries_per_class`
- `model.py`: mean-pool text encoder plus prototypical classifier head
- `train.py`: episodic training loop and checkpointed smoke-friendly runner

## Quick Run

```bash
python -m tracks.nlp.lesson_11_toy_few_shot_text_classification.train \
  --epochs 1 \
  --num-episodes 48 \
  --batch-size 4 \
  --num-ways 3 \
  --shots 2 \
  --queries-per-class 2 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Expected Outputs

Runs write to `outputs/nlp/lesson_11_toy_few_shot_text_classification/<run_name>/` and produce:

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `checkpoints/checkpoint.pt`
