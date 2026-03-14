# Lesson 10: OWL-ViT-Lite Toy Open-Vocabulary Detection

This lesson introduces text-conditioned open-vocabulary detection:

- render scenes with multiple colored shapes
- query the image with text like `detect red square`
- allow the query to be present or absent
- predict both presence and location from one text-conditioned detector

## Run

```bash
python -m tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_10_owlvit_toy_open_vocab_detection/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add synonym-heavy queries so the detector must handle more than one verb or attribute phrase.
2. Replace the single-query setup with several candidate queries per image and compare batching complexity.
3. Extend the box head into a mask head and compare this lesson directly against lesson 5.
