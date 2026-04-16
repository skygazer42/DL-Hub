# Lesson 27: Toy Self-Correction Prompting

This lesson demonstrates a tiny self-correction prompting template on synthetic sequences.

Each sample contains:

- a `PROMPT` token and topic token
- a `DRAFT` span with an intentionally weak draft answer
- a `CRITIQUE` span describing the draft issue
- a `CORRECT` marker followed by the corrected answer tokens

Training uses response-only masking for the correction span: prompt, draft, and critique traces
are context only, while corrected answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_27_toy_self_correction_prompting.train --device cpu --epochs 1
```
