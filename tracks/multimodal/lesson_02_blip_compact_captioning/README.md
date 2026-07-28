# Lesson 02: BLIP-Lite Captioning And ITM

This lesson upgrades the multimodal track from dual-encoder alignment to fused generation:

- encode an image into visual tokens
- decode a sentence while attending to those visual tokens
- classify whether an image and caption match
- optimize captioning and ITM together

## Run

```bash
python -m tracks.multimodal.lesson_02_blip_compact_captioning.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_02_blip_compact_captioning/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Raise `--negative-fraction` and see how ITM accuracy and caption quality trade off.
2. Replace GRU decoding with a tiny Transformer decoder.
3. Add harder negatives that only change one attribute, such as color or location.
