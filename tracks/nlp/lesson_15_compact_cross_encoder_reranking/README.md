# Lesson 15: Compact Cross-Encoder Reranking

This lesson scores query/document pairs and learns to rank a relevant document above a sampled
negative. Synthetic concepts cover fruit, vehicles, animals, and instruments; each example is a
query, positive passage, and passage drawn from another concept.

## Implementation

- `data.py` tokenizes each query and document into one sequence separated by `<sep>`.
- `model.py` applies positional embeddings and compact Transformer encoder blocks, mean-pools the
  joint sequence, and produces one relevance score.
- `train.py` uses pairwise ranking loss and reports the fraction of positive scores above negatives.

## Quick Run

```bash
python -m tracks.nlp.lesson_15_compact_cross_encoder_reranking.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_15_compact_cross_encoder_reranking/<run_name>/`. A
successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite train/eval losses and rerank accuracies in
`[0, 1]` under `train_rerank_acc` and `eval_rerank_acc`.
