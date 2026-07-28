# Lesson 34: Video-Text Compact Action Recognition

This lesson introduces compact video-language action recognition:

- synthesize short video clip features with one dominant action
- pair each clip with a text description query
- fuse temporal video features and text context
- classify the action label (`jump`, `wave`, `sit`)

## Run

```bash
python -m tracks.multimodal.lesson_34_video_text_action_recognition.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_34_video_text_action_recognition/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
