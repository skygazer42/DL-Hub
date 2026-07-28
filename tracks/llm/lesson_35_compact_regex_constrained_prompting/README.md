# Lesson 35: Compact Regex-Constrained Prompting

This lesson teaches regex-constrained output continuation with a tiny synthetic causal LM task.
Each example provides pattern hints for literals, a character class, and a quantifier, then starts
supervision at an explicit `REGEX` marker so the model learns only the regex continuation.

Run:

```bash
python -m tracks.llm.lesson_35_compact_regex_constrained_prompting.train --device cpu --epochs 1
```
