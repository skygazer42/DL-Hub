# Lesson 07: Q-Former-Lite Toy Bridge VLM

This lesson introduces a query bottleneck between vision and language:

- encode the image into spatial visual tokens
- let a small set of learned query tokens read from those visual tokens
- pass only the query states into a tiny decoder LM
- answer a short visual question

## Run

```bash
python -m tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_07_qformer_toy_bridge_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Increase `num_query_tokens` and compare exact match versus runtime.
2. Add another query block and inspect whether attention becomes sharper.
3. Compare this lesson directly against lesson 3 on the same QA data.
