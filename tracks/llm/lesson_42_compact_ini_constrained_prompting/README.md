# Lesson 42: Compact INI Constrained Prompting

This lesson trains a tiny causal Transformer to emit a structured INI suffix after a compact prompt.
Each sequence switches into the supervised region at `ini_token_id`, and prompt-side labels remain
masked with `ignore_index`.

## Run

```bash
python -m tracks.llm.lesson_42_compact_ini_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written to `outputs/llm/lesson_42_compact_ini_constrained_prompting/<run_name>/`.
