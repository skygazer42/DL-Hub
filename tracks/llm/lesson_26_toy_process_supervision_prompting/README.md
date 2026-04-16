# Lesson 26: Toy Process Supervision Prompting

This lesson demonstrates a compact process-supervision setup for language models. Each
synthetic sequence contains a prompt, an imperfect draft, explicit process checks, and a
final `PROCESS` marker that asks the model to generate the corrected answer span.

The lesson covers:
- response-only token masking that starts at the `PROCESS` marker
- a small causal transformer language model
- lightweight training with config, vocab, metrics, samples, and checkpoint artifacts

Run locally with:

```bash
python -m tracks.llm.lesson_26_toy_process_supervision_prompting.train --device cpu --epochs 1
```
