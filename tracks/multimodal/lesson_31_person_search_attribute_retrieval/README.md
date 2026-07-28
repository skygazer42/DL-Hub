# Lesson 31: Person-Search Attribute Retrieval (Compact)

This lesson maps person-search to a tiny image-text retrieval setup inspired by Re-ID:
synthetic person images are paired with short attribute queries like
`person red black backpack`.

## What it demonstrates

- Synthetic person profile generation (`shirt`, `pants`, `accessory`)
- Attribute-text query encoding with a tiny vocabulary
- CLIP-style contrastive alignment between person images and text queries
- Retrieval metrics (`top-1` and `recall@3`) on CPU in a smoke-sized run

## Run

```bash
python -m tracks.multimodal.lesson_31_person_search_attribute_retrieval.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1
```

## Outputs

`outputs/multimodal/lesson_31_person_search_attribute_retrieval/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Notes

- This is intentionally compact-first for data flow and retrieval mechanics.
- It is not a production Re-ID system and uses simple synthetic visuals.
