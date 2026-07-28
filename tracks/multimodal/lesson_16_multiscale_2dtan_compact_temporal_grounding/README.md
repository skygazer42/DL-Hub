# Lesson 16: Multi-Scale 2D-TAN-Lite Compact Temporal Grounding

This lesson extends lesson 15 from a single dense temporal segment map to multi-scale temporal grounding:

- render a short video with one object and one target event segment
- ask when the queried event happens
- build dense upper-triangular segment maps at three temporal scales
- fuse coarse and fine temporal predictions into one final segment prediction

## Run

```bash
python -m tracks.multimodal.lesson_16_multiscale_2dtan_compact_temporal_grounding.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_16_multiscale_2dtan_compact_temporal_grounding/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
