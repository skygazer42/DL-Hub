# Lesson 22: Compact Audio-Visual Event Localization

This lesson builds a compact multimodal localizer that answers when a queried event happens:

- render short compact video clips with a single salient event frame
- render aligned per-frame audio clips where the event frame is strongest
- encode video, audio, and text query
- fuse all three modalities per frame
- predict the event timestamp in the clip

## Run

```bash
python -m tracks.multimodal.lesson_22_audio_visual_event_localization.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_22_audio_visual_event_localization/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
