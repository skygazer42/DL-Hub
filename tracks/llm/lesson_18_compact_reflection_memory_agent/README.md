# Lesson 18: Compact Reflection Memory Agent

This lesson builds a tiny synthetic reflection-memory agent workflow for ChatGPT-style training.

Each sequence contains:

- a user query and an initial assistant draft
- a reflection step that identifies a correction
- a memory write and memory read marker
- a revision turn where the assistant rewrites the answer using retrieved memory

Only revision answer tokens are supervised (assistant-only masked loss over the revised span).

Run:

```bash
python -m tracks.llm.lesson_18_compact_reflection_memory_agent.train --device cpu --epochs 1
```
