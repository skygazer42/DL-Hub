# Lesson 17: Compact Text Anomaly Detection

This lesson separates ordinary procedural sentences from intentionally incoherent token mixtures.
The `--anomaly-fraction` option controls class balance, and every sample is generated from local
templates for deterministic, network-free experiments.

## Implementation

- `data.py` mixes normal and anomalous templates, builds a local vocabulary, and creates a seeded split.
- `model.py` mean-pools token embeddings and applies a small MLP binary classifier.
- `train.py` uses binary cross-entropy with logits and reports anomaly classification accuracy.

## Quick Run

```bash
python -m tracks.nlp.lesson_17_compact_text_anomaly_detection.train \
  --epochs 1 --num-samples 128 --batch-size 16 --anomaly-fraction 0.35 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/nlp/lesson_17_compact_text_anomaly_detection/<run_name>/`. A
successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite loss and `train_anomaly_acc` plus
`eval_anomaly_acc` values in `[0, 1]`.
