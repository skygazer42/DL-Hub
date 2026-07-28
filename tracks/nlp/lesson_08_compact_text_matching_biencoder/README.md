# Lesson 08: Compact Text Matching Bi-Encoder

This lesson introduces a minimal text matching and retrieval setup:

- encode a query and a document with one shared text encoder
- score aligned pairs with cosine-style similarity
- optimize both pair matching and in-batch retrieval

The data is synthetic and CPU friendly. Each batch is built from aligned query/document pairs so
the diagonal of the similarity matrix is the retrieval target.

## Run

```bash
python -m tracks.nlp.lesson_08_compact_text_matching_biencoder.train \
  --device cpu --epochs 2 --max-train-batches 5 --max-eval-batches 5 --run-name smoke
```

## Outputs

`outputs/nlp/lesson_08_compact_text_matching_biencoder/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
