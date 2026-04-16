# Lesson 07: Toy Reward Modeling

This lesson adds a tiny reward model after SFT/preference optimization style data prep.

The toy dataset emits `(prompt, chosen_completion, rejected_completion)` triples where the chosen completion follows a deterministic "better" pattern and the rejected completion follows a deterministic "worse" pattern.

Training objective:

- score chosen and rejected sequences with a scalar reward model
- optimize pairwise ranking loss so `reward(chosen) > reward(rejected)`

Run:

```bash
python -m tracks.llm.lesson_07_toy_reward_modeling.train --device cpu --epochs 1
```
