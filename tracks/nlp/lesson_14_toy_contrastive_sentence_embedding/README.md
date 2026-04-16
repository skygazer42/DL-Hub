# Lesson 14: Toy Contrastive Sentence Embedding

This lesson introduces sentence representation learning with two augmented views and an NT-Xent style objective.

## What You Learn

- how to build two stochastic sentence views with token dropout/deletion
- how an encoder plus projection head can map sentence views into a shared space
- how to optimize paired similarities with a contrastive objective

## Files

- `data.py`: synthetic sentence generation, view augmentation, and dataloaders
- `model.py`: sentence encoder + projection head + contrastive utilities
- `train.py`: training loop, metrics logging, and checkpointing

## Quick Run

```bash
python -m tracks.nlp.lesson_14_toy_contrastive_sentence_embedding.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --max-length 10 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Expected Outputs

Runs write to `outputs/nlp/lesson_14_toy_contrastive_sentence_embedding/<run_name>/` and produce:

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `checkpoints/checkpoint.pt`
