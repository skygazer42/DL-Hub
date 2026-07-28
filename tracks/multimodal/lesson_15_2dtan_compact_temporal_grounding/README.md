# Lesson 15: 2D-TAN-Lite Compact Temporal Grounding

This lesson will introduce text-conditioned temporal grounding with a dense 2D segment map:

- render a short video with one object and one target event segment
- ask when the queried event happens
- build a dense upper-triangular temporal map
- score every valid segment cell directly

## Run

```bash
python -m tracks.multimodal.lesson_15_2dtan_compact_temporal_grounding.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_15_2dtan_compact_temporal_grounding/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
