# Lesson 24: Compact Multimodal Reasoning

This lesson answers a binary question by combining an image cue with textual facts. A red or blue
image selects candidate A or B, the facts assign each candidate a material, and the query asks
whether the selected candidate has a requested material. All images and text are generated offline.

## Implementation

- `data.py` emits a color-patch image, tokenized candidate facts, a material query, and a yes/no label.
- `model.py` joins a tiny CNN feature with separately pooled fact and query embeddings, then predicts
  two answer classes through an MLP fusion head.
- `train.py` optimizes cross-entropy and reports loss and answer accuracy for both splits.

## Quick Run

```bash
python -m tracks.multimodal.lesson_24_multimodal_reasoning.train \
  --epochs 1 --num-samples 64 --batch-size 8 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/multimodal/lesson_24_multimodal_reasoning/<run_name>/`. A successful
run writes `config.json`, `metrics.jsonl`, `logs/train.log`, and `checkpoints/checkpoint.pt`.
Acceptance requires finite train/eval losses and `train_accuracy` plus `eval_accuracy` values in the
closed interval `[0, 1]`.
