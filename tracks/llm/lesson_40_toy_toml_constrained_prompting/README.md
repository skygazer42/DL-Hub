# Lesson 40: Toy TOML Constrained Prompting

This lesson trains a tiny causal Transformer to emit a structured TOML suffix after a toy prompt.
Each sequence switches into the supervised region at `toml_token_id`, and prompt-side labels stay
masked with `ignore_index`.

## Run

```bash
python -m tracks.llm.lesson_40_toy_toml_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written to `outputs/llm/lesson_40_toy_toml_constrained_prompting/<run_name>/`.
