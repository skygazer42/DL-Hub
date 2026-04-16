# Lesson 37: Toy SQL Constrained Prompting

This lesson teaches structured continuation under a simple SQL-like output constraint. Each
synthetic sequence provides a compact prompt plus a tiny schema header (table + column tokens),
then switches at an explicit `sql_token_id` marker into the supervised query region.

All prompt-side tokens are masked to `ignore_index` so training starts exactly at the SQL boundary.
The model is a small causal transformer LM that predicts the SQL tail and writes standard lesson
artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_37_toy_sql_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/llm/lesson_37_toy_sql_constrained_prompting/<run_name>/`.

