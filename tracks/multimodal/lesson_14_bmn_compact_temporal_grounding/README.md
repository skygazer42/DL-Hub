# Lesson 14: BMN-Lite Compact Temporal Grounding

This lesson will introduce text-conditioned temporal grounding over short videos:

- render a short video with one object and one target event segment
- ask when the queried event happens
- predict start and end boundaries
- score an upper-triangular proposal map

## Run

```bash
python -m tracks.multimodal.lesson_14_bmn_compact_temporal_grounding.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_14_bmn_compact_temporal_grounding/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
