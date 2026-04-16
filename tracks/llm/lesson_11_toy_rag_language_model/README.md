# Lesson 11: Toy RAG Language Model

This lesson implements a tiny retrieval-augmented language model on synthetic data.

Each sample:

- creates a short query sequence
- retrieves one synthetic document id (`doc_id`)
- conditions the decoder on that document embedding
- predicts the next token with a causal LM loss (`ignore_index=-100` on pads)

Run:

```bash
python -m tracks.llm.lesson_11_toy_rag_language_model.train --device cpu --epochs 1
```
