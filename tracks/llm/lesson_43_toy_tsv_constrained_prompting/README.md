# Lesson 43: Toy TSV Constrained Prompting

This lesson teaches structured continuation under a tiny TSV output constraint. Each synthetic
sequence provides a compact prompt plus column hints, then switches at an explicit `tsv_token_id`
marker into the supervised tab-separated region.

All prompt-side tokens are masked to `ignore_index` so training starts exactly at the TSV
boundary. The model is a small causal transformer LM that predicts the table tail and writes the
standard lesson artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_43_toy_tsv_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under `outputs/llm/lesson_43_toy_tsv_constrained_prompting/<run_name>/`.
