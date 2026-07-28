# Lesson 36: Compact EBNF Constrained Prompting

This lesson teaches structured continuation under a tiny EBNF-style output constraint. Each
synthetic sequence begins with prompt-side grammar metadata, then switches at an explicit
`ebnf_token_id` marker into the supervised EBNF payload region.

The dataset masks all prefix tokens to `ignore_index` so training begins exactly at the constrained
generation boundary. A small causal transformer predicts the structured suffix and writes the
standard lesson artifacts for smoke testing.

Run:

```bash
python -m tracks.llm.lesson_36_compact_ebnf_constrained_prompting.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/llm/lesson_36_compact_ebnf_constrained_prompting/<run_name>/`.
