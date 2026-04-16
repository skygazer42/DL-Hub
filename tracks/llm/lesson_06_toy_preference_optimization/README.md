# Lesson 06: Toy Preference Optimization

This lesson adds a tiny pairwise preference objective after instruction/prefix tuning.

What changes relative to earlier SFT lessons:

- each sample contains a prompt plus two candidate continuations: `chosen` and `rejected`
- training optimizes a DPO-style pairwise loss against a frozen reference model
- the dataset remains synthetic, deterministic, and CPU-friendly

Run:

```bash
python -m tracks.llm.lesson_06_toy_preference_optimization.train --device cpu --epochs 1
```
