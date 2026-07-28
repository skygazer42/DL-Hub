# Lesson 29: Compact Constraint-Repair Prompting

This lesson demonstrates a tiny constraint-repair prompting template on synthetic sequences.

Each sample contains:

- a `PROMPT` token and topic token
- a `CONSTRAINT` span that specifies a simple target pattern
- a `CANDIDATE` span with an answer that violates the constraint
- a `REPAIR` marker followed by the corrected answer tokens

Training uses response-only masking for the repair span: prompt, constraint, and candidate
traces are context only, while repaired answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_29_compact_constraint_repair_prompting.train --device cpu --epochs 1
```
