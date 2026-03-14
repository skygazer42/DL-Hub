# Lesson 13: Video-VLM-Lite Toy Temporal QA

This lesson introduces prompt-conditioned temporal QA over short videos:

- render a short video with one moving colored shape
- ask about color, shape, or motion direction
- aggregate frame features over time
- generate a short answer token such as `red`, `circle`, `yes`, or `no`

## Run

```bash
python -m tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_13_video_vlm_toy_temporal_qa/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add a speed question and compare whether temporal pooling is still enough.
2. Increase the sequence length and compare mean pooling against recurrent aggregation.
3. Extend the lesson to two objects and keep the same question interface.
