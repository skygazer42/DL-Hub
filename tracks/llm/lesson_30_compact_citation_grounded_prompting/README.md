# Lesson 30: Compact Citation-Grounded Prompting

This lesson builds a tiny causal language-model task where the answer must include a compact
citation copied from an explicit reference span. Each synthetic sequence contains:

- a `PROMPT` token with a topic token
- a `REFERENCE` span that provides the evidence tokens
- an `ANSWER` span with a short free-form response stub
- a `CITE` marker followed by the supervised citation tokens and EOS

Training uses response-only masking for the citation span. Prompt, reference, and answer stub
tokens are context only, while cited evidence tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_30_compact_citation_grounded_prompting.train --device cpu --epochs 1
```
