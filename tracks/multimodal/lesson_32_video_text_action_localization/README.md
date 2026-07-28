# Lesson 32: Video-Text Compact Temporal Action Localization (时序动作定位)

This lesson introduces compact video-text temporal action localization:

- synthesize short per-frame video features and a text query
- localize the query action over time with a temporal mask
- decode the predicted start/end segment from the mask
- train and evaluate with temporal IoU and Recall@IoU

## Run

```bash
python -m tracks.multimodal.lesson_32_video_text_action_localization.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_32_video_text_action_localization/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
