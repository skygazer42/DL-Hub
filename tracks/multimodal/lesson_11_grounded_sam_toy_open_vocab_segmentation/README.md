# Lesson 11: Grounded-SAM-Lite Toy Open-Vocabulary Segmentation

This lesson introduces text-conditioned open-vocabulary segmentation:

- render scenes with multiple colored shapes
- query the image with text like `segment red square`
- allow the query to be present or absent
- predict both presence and a low-resolution foreground mask

## Run

```bash
python -m tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add more query verbs and compare whether the prompt encoder stays robust.
2. Replace single-query supervision with several candidate queries per image.
3. Add a point prompt and compare this lesson against a more SAM-like interactive setting.
