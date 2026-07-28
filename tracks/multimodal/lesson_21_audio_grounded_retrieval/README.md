# Lesson 21: Compact Audio-Grounded Retrieval

This lesson builds a compact multimodal retrieval setup where language queries are grounded in paired audio and video evidence.

- synthesize short clips with moving shapes and aligned compact spectrograms
- generate segment-aware text queries (intro/middle/outro)
- encode video and audio with tiny towers, then fuse them into clip embeddings
- encode text queries and align clips/queries with a symmetric contrastive loss

The objective is to retrieve the matching clip segment from a query that references both modalities.

## Run

```bash
python -m tracks.multimodal.lesson_21_audio_grounded_retrieval.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_21_audio_grounded_retrieval/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Replace average temporal pooling with a tiny segment-attention module.
2. Add hard-negative mining by shuffling segment ids within each event.
3. Compare retrieval accuracy with audio-only, video-only, and fused evidence.
