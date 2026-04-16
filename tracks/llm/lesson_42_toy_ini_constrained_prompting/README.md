# Lesson 42: Toy INI Constrained Prompting

This lesson trains a tiny causal Transformer to emit a structured INI suffix after a toy prompt.
Each sequence switches into the supervised region at `ini_token_id`, and prompt-side labels remain
masked with `ignore_index`.

## Run

```bash
python -m tracks.llm.lesson_42_toy_ini_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written to `outputs/llm/lesson_42_toy_ini_constrained_prompting/<run_name>/`.
