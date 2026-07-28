# Lesson 41: Compact Markdown-Table Constrained Prompting

This lesson teaches structured continuation under a tiny markdown-table output constraint. Each
synthetic sequence provides a compact prompt plus column hints, then switches at an explicit
`markdown_table_token_id` marker into the supervised table region.

All prompt-side tokens are masked to `ignore_index` so training starts exactly at the markdown
table boundary. The model is a small causal transformer LM that predicts the table tail and writes
standard lesson artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_41_compact_markdown_table_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/llm/lesson_41_compact_markdown_table_constrained_prompting/<run_name>/`.
