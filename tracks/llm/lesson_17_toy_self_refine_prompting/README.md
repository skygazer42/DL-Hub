# Lesson 17: Toy Self-Refine Prompting

This lesson adds a tiny self-refine supervision pattern on synthetic token sequences.

Each sample has:

- a prompt token/value
- an initial draft span
- a critique span
- a refine marker followed by the refined answer

Only refined-answer tokens (plus EOS) are supervised in the loss, while prompt, draft, and critique context are masked out.

Run:

```bash
python -m tracks.llm.lesson_17_toy_self_refine_prompting.train --device cpu --epochs 1
```
