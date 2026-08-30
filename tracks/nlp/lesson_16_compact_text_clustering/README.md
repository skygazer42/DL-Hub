# Lesson 16: Compact Text Clustering

This lesson maps short synthetic sentences into four semantic groups: science, sports, finance,
and art. It is a supervised cluster-assignment exercise using known synthetic cluster IDs, not an
unsupervised k-means implementation; the projected embeddings remain available for exploration.

## Implementation

- `data.py` samples topic templates, tokenizes them, and supplies seeded train/validation splits.
- `model.py` mean-pools token embeddings, projects them to a compact representation, and predicts a
  cluster logit vector.
- `train.py` optimizes cross-entropy against the synthetic cluster IDs and records assignment accuracy.

## Quick Run

```bash
python -m tracks.nlp.lesson_16_compact_text_clustering.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_16_compact_text_clustering/<run_name>/`. A successful run
writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite loss and `train_cluster_acc` plus
`eval_cluster_acc` values in `[0, 1]`.
