# Lesson 25: Compact Verifier-Guided Prompting

This lesson demonstrates a tiny verifier-guided prompting template on synthetic sequences.

Each sample contains:

- a `PROMPT` token and topic token
- two `CANDIDATE` spans with short candidate tokens
- a verifier stage (`VERIFY` + pass/fail marker) after each candidate
- a `GUIDE` marker followed by the final verifier-approved answer tokens

Training uses response-only masking for the guided answer span: candidate/verifier traces are
context only, while guided answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_25_compact_verifier_guided_prompting.train --device cpu --epochs 1
```
