# Lesson 22: Toy Self-Consistency Prompting

This lesson demonstrates a tiny self-consistency prompting pattern on synthetic sequences.

Each example contains:

- a prompt token/value
- three sampled candidate answer traces
- lightweight agreement or disagreement markers per sample
- a `vote` stage that emits the final majority-consistent answer tokens

Training supervises only the voting stage, so the sampled traces remain context while the final
consistent answer path is learned autoregressively.

Run:

```bash
python -m tracks.llm.lesson_22_toy_self_consistency_prompting.train --device cpu --epochs 1
```
