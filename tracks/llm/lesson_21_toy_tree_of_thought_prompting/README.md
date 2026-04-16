# Lesson 21: Toy Tree-of-Thought Prompting

This lesson demonstrates a tiny tree-of-thought prompting pattern on synthetic sequences.

Each example contains:

- a prompt token/value
- two candidate reasoning branches
- lightweight branch quality markers
- a `choose` stage that emits the final answer tokens from the better branch

Training supervises only the final selection stage, so the branch traces remain context while the
chosen answer path is learned autoregressively.

Run:

```bash
python -m tracks.llm.lesson_21_toy_tree_of_thought_prompting.train --device cpu --epochs 1
```
