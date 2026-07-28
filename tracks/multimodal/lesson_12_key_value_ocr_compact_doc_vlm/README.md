# Lesson 12: Key-Value OCR-Lite Compact Document VLM

This lesson introduces prompt-conditioned document OCR:

- render a compact document image with several `key:value` rows
- query the document with prompts like `read total`
- generate the requested value as text
- return `none` when the requested field is missing

## Run

```bash
python -m tracks.multimodal.lesson_12_key_value_ocr_compact_doc_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add more field types and compare missing-field accuracy.
2. Split dates or ids into multiple output tokens instead of one token.
3. Compare this lesson directly against lesson 9 on the same decoder architecture.
