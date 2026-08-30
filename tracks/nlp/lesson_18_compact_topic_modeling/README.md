# Lesson 18: Compact Topic Modeling

This lesson learns topic mixtures for synthetic science, sports, finance, and travel sentences.
Topic labels are retained only to measure alignment; the optimization target reconstructs each
sentence's bag-of-words vector with a small entropy regularizer on topic probabilities.

## Implementation

- `data.py` returns token IDs, attention masks, binary bag-of-words targets, and held-out topic labels.
- `model.py` infers topic probabilities from pooled embeddings and reconstructs vocabulary logits
  through a learned topic-word matrix.
- `train.py` reports total loss, reconstruction loss, and label-based topic accuracy.

## Quick Run

```bash
python -m tracks.nlp.lesson_18_compact_topic_modeling.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_18_compact_topic_modeling/<run_name>/`. A successful run
writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite total/reconstruction losses and train/eval
topic accuracies in `[0, 1]`.
