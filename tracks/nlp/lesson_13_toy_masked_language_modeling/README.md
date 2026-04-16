# Lesson 13: Toy Masked Language Modeling

This lesson introduces a tiny self-supervised masked language modeling (MLM) task.
Given synthetic sentences, the dataloader replaces a subset of input tokens with
`<mask>` and trains a small transformer encoder to recover the original tokens.

## What You Learn

- How to prepare token-level supervision with `ignore_index=-100`.
- How MLM differs from supervised classification in the same NLP track.
- How to report masked-token accuracy during training and evaluation.

## Run

From the repository root:

```bash
python -m tracks.nlp.lesson_13_toy_masked_language_modeling.train \
  --device cpu \
  --epochs 2 \
  --num-samples 512 \
  --batch-size 32 \
  --max-length 12 \
  --mask-prob 0.15
```

Outputs are written under:

`outputs/nlp/lesson_13_toy_masked_language_modeling/<run_name>/`
