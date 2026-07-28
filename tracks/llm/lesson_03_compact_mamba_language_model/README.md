# Lesson 03: compact Mamba / state-space language model

This lesson keeps the same synthetic next-token task as the transformer LM, but swaps causal self-attention for a tiny recurrent selective state-space mixer.

- Input: fixed-length `input_ids`
- Target: predict the next token at every position
- Mixer: a simplified selective scan that updates a latent state token by token
- Generation: greedy autoregressive decoding from a short prompt

The implementation is intentionally small and CPU-friendly. It is not a faithful Mamba reproduction; it is a teaching example that shows how causal sequence mixing can happen through a learned recurrent state instead of attention.

## Run

```bash
python -m tracks.llm.lesson_03_compact_mamba_language_model.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## Outputs

`outputs/llm/lesson_03_compact_mamba_language_model/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
