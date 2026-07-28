# Lesson 09: PaliGemma-Lite Compact SigLIP Decoder VLM

This lesson introduces prompt-native multitask text generation:

- render one synthetic object image
- switch between caption, attribute, locate, and yes/no prompts
- encode the image with a small SigLIP-style vision tower
- predict every answer as text with one decoder-only LM

## Run

```bash
python -m tracks.multimodal.lesson_09_paligemma_compact_siglip_decoder_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_09_paligemma_compact_siglip_decoder_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add a `size` prompt and compare exact match.
2. Add a second object and keep the text interface unchanged.
3. Compare this lesson directly against lesson 3 on the same QA subset.
