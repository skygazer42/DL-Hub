# Lesson 20: Compact Audio-Visual Learning

This lesson teaches a compact audio-visual learning loop on synthetic clips:

- render a short moving-shape video sequence
- render a paired compact spectrogram with an aligned temporal pattern
- encode video and audio with separate tiny towers
- align the two modalities with a symmetric contrastive objective
- fuse both streams to predict the underlying event and motion labels

## Run

```bash
python -m tracks.multimodal.lesson_20_audio_visual_learning.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_20_audio_visual_learning/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Replace mean video pooling with a tiny temporal attention block.
2. Add mismatched audio negatives inside each batch and compare retrieval accuracy.
3. Predict a binary sync / out-of-sync label by time-shifting the spectrogram.
