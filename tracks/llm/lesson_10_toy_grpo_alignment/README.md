# Lesson 10: Toy GRPO Alignment

This lesson implements a tiny, CPU-friendly version of group-relative policy optimization (GRPO).

- Synthetic prompts are paired with small response groups.
- Each candidate in a group has a scalar reward rank.
- The policy is trained with a simple grouped objective that centers rewards within each group and
  weights response-token log-probability accordingly.

Run:

```bash
python -m tracks.llm.lesson_10_toy_grpo_alignment.train --device cpu --epochs 1 --max-train-batches 4 --max-eval-batches 2
```
