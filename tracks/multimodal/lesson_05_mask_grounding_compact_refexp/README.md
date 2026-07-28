# Lesson 05: Mask-Grounding-Lite Compact Referring Expressions

This lesson extends text-conditioned grounding from boxes to low-resolution masks:

- render scenes with multiple objects
- encode a referring expression
- fuse text features into a spatial visual map
- predict a low-resolution foreground mask for the target object

## Run

```bash
python -m tracks.multimodal.lesson_05_mask_grounding_compact_refexp.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_05_mask_grounding_compact_refexp/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add harder referring expressions that use relative position between objects.
2. Change `mask_size` and compare IoU versus runtime.
3. Extend the lesson to support multiple target masks in one image.
