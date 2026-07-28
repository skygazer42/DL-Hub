# Lesson 38: Compact YAML Constrained Prompting

This lesson extends the structured-output prompting block with a YAML-style decoder target. Each
sequence contains an instruction prefix and then switches at an explicit `yaml_token_id` marker into
the supervised YAML region.

The compact task keeps the setup tiny: the model predicts two key-value lines after the marker, and the
prompt-side labels are masked with `ignore_index` so only the constrained output segment contributes
to the loss.

## Run

```bash
python -m tracks.llm.lesson_38_compact_yaml_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written to `outputs/llm/lesson_38_compact_yaml_constrained_prompting/<run_name>/`.
