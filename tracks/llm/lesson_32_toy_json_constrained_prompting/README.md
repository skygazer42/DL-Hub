# Lesson 32: Toy JSON-Constrained Prompting

This lesson builds a tiny causal language-model task where the answer must continue as a compact
JSON-style record after an explicit `JSON` marker.

Each synthetic sequence contains:

- a `PROMPT` token followed by a topic token
- short field hints that describe the required keys
- a `JSON` marker that starts the supervised structured continuation
- compact key/value tokens followed by EOS

Training uses response-only masking from the `JSON` marker onward so prompt and field hints remain
context-only while the structured output tokens are supervised.

Run:

```bash
python -m tracks.llm.lesson_32_toy_json_constrained_prompting.train --device cpu --epochs 1
```
