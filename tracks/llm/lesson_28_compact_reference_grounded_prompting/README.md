# Lesson 28: Compact Reference-Grounded Prompting

This lesson builds a tiny causal language-model task where the answer must be grounded in an
explicit reference span. Each synthetic sequence contains:

- a `PROMPT` token with a topic token
- a `REFERENCE` span that supplies usable evidence
- a `QUERY` span that asks for the grounded answer
- a `GROUND` marker followed by the supervised grounded response tokens

Training uses response-only masking for the grounded answer span. Prompt, reference, and query
tokens are context only, while grounded response tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_28_compact_reference_grounded_prompting.train --device cpu --epochs 1
```
