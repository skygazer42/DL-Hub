# Lesson 06: Flamingo-Lite Toy Interleaved VLM

This lesson introduces interleaved image-text prompting:

- build a prompt with support examples and a query
- place `<image>` markers inside the text stream
- align one image embedding to each image marker
- predict the query answer from the full multimodal prompt

## Run

```bash
python -m tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_06_flamingo_toy_interleaved_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add more support shots and compare exact match.
2. Replace image-slot injection with a small gated cross-attention block.
3. Add distractor support examples with inconsistent task words.
