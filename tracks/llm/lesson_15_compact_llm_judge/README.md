# Lesson 15: Compact LLM Judge

This lesson implements a tiny LLM-judge style objective on synthetic data.

Each sample contains:

- a prompt token/value
- a reference answer signal token/value
- a candidate answer token/value
- a verdict token (`good` or `bad`)

The model learns:

- next-token prediction over the synthetic sequence
- a scalar judge score for candidate quality

Run:

```bash
python -m tracks.llm.lesson_15_compact_llm_judge.train --device cpu --epochs 1
```
