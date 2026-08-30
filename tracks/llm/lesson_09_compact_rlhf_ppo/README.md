# Lesson 09: Compact RLHF PPO

This lesson demonstrates the mechanics of a clipped policy objective with a frozen reference
language model and token-level synthetic rewards. Prompt/response sequences are generated locally;
responses carry a `good` or `bad` marker that the deterministic reward model converts to a score.

This is deliberately a PPO-style teaching approximation, not full online RLHF: batches come from a
fixed dataset and the frozen initialization acts as both old and reference policy. There are no
on-policy rollouts or separately trained reward model.

## Implementation

- `data.py` builds padded prompt/response sequences and masks optimization to response tokens.
- `model.py` reuses the compact causal Transformer policy and adds a marker-based token reward model.
- `train.py` applies clipped log-probability ratios, normalized rewards, and a reference KL penalty.

## Quick Run

```bash
python -m tracks.llm.lesson_09_compact_rlhf_ppo.train \
  --epochs 1 --num-samples 128 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/llm/lesson_09_compact_rlhf_ppo/<run_name>/`. A successful run writes
`config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite `train_loss` and `eval_loss` entries;
`samples.jsonl` records an input, target labels, synthetic reward, and policy loss for inspection.
