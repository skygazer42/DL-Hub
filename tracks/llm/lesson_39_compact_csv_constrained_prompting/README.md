# Lesson 39: Compact CSV Constrained Prompting

This lesson teaches structured continuation under a simple CSV-like output constraint. Each
synthetic sequence provides a compact prompt plus a tiny header (column names), then switches at
an explicit `csv_token_id` marker into the supervised table region.

All prompt-side tokens are masked to `ignore_index` so training starts exactly at the CSV boundary.
The model is a small causal transformer LM that predicts the CSV tail and writes standard lesson
artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_39_compact_csv_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/llm/lesson_39_compact_csv_constrained_prompting/<run_name>/`.
