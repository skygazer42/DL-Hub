# Lesson 26: Image-Text Reranking (Compact)

This lesson builds a tiny cross-modal reranker for a single image and a small list of
candidate text descriptions.

## What it demonstrates

- Synthetic image generation from compact visual concepts (`color`, `texture`, `shape`)
- Candidate-set reranking with one positive caption and sampled hard negatives
- CPU-friendly training loop with `cross_entropy` over candidate scores

## Run

```bash
python -m tracks.multimodal.lesson_26_image_text_reranking.train --device cpu --epochs 1
```

## Notes

- The model is intentionally small: two tiny encoders and a shallow scorer MLP.
- This is a compact lesson for data flow and training structure, not production retrieval quality.
