# Lesson 51: Compact World Models

This lesson builds a minimal world-model training loop over synthetic transitions.

## What it covers

- synthetic transition tuples `(obs, action, prompt) -> (next_obs, reward, done)`
- direct use of local world-model families from `dlhub.generative.world_models`
- compact multi-head supervision: reconstruction, reward regression, done prediction
- CPU-friendly smoke training with standard run artifacts

## Run

```bash
python -m tracks.generative.lesson_51_compact_world_models.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

