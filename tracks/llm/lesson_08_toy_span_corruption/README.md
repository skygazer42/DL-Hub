# Lesson 08: Toy Span Corruption

This lesson implements a tiny span-corruption objective with a decoder-only Transformer.

Each synthetic sample:

- draws a random content sequence
- replaces one contiguous span with a `<mask>` token
- appends the removed span after a target delimiter
- computes loss only on appended target tokens (other positions use `-100`)

Run:

```bash
python -m tracks.llm.lesson_08_toy_span_corruption.train --device cpu --epochs 1
```
