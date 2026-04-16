# Lesson 24: Toy Debate Prompting

This lesson demonstrates a compact debate-style prompting template on synthetic token sequences.
Each example contains a `PROMPT`, a `CLAIM`, short `PRO` and `CON` argument spans, and a `JUDGE`
marker followed by the supervised verdict tokens.

Training uses response-only masking for the verdict span: the prompt, claim, and argument traces
are context, while the post-`JUDGE` answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_24_toy_debate_prompting.train --device cpu --epochs 1
```
