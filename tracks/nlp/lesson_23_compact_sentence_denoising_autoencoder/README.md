# Lesson 23: Compact Sentence Denoising Autoencoder

This lesson corrupts short synthetic sentences with masking, deletion, token replacement, and
small swaps, then trains a compact seq2seq model to reconstruct the clean text.

Quick smoke run:

```bash
python -m tracks.nlp.lesson_23_compact_sentence_denoising_autoencoder.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs land under `outputs/nlp/lesson_23_compact_sentence_denoising_autoencoder/<run_name>/`.
