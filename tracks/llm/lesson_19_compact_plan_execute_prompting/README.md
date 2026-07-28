# Lesson 19: Compact Plan-Execute Prompting

This lesson demonstrates a tiny two-stage prompting setup on synthetic sequences.

Each sample contains:

- a prompt token/value
- a plan stage with short intermediate tokens
- an execute marker followed by the final answer tokens

Training uses response-only masking for the execute stage: prompt and plan tokens are context only, while execute answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_19_compact_plan_execute_prompting.train --device cpu --epochs 1
```
