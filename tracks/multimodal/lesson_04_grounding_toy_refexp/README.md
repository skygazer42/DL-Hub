# Lesson 04: Grounding-Lite Toy Referring Expressions

This lesson introduces text-conditioned spatial grounding:

- render scenes with multiple objects
- encode a referring expression
- predict the target grid cell
- decode a bounding box from the selected cell plus local offsets

## Run

```bash
python -m tracks.multimodal.lesson_04_grounding_toy_refexp.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_04_grounding_toy_refexp/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add harder expressions that omit one attribute and rely more on location.
2. Change the grid size and inspect the tradeoff between cell accuracy and box error.
3. Extend the head to support multiple targets per image.
