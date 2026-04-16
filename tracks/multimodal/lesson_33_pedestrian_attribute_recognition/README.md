# Lesson 33: Pedestrian Attribute Recognition (Toy)

This lesson teaches a small image-text retrieval setup for pedestrian attributes.
A synthetic pedestrian image is paired with a compact attribute query such as
`pedestrian red hoodie backpack`.

## What it demonstrates

- Synthetic pedestrian image generation from attributes (`color`, `upper-wear`, `accessory`)
- Text-query encoding with a compact vocabulary
- CLIP-style contrastive alignment between pedestrian images and attribute phrases
- CPU-friendly retrieval metrics (`top-1` and `recall@3`)

## Run

```bash
python -m tracks.multimodal.lesson_33_pedestrian_attribute_recognition.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1
```

## Outputs

`outputs/multimodal/lesson_33_pedestrian_attribute_recognition/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Notes

- This lesson is toy-first for understanding multimodal attribute prediction flow.
- It is not intended as a production pedestrian attribute recognition system.
