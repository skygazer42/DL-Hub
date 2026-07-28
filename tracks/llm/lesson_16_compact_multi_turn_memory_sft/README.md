# Lesson 16: Compact Multi-Turn Memory Chat SFT

This lesson implements a tiny synthetic multi-turn chat dataset where the model must use prior dialogue context.

Each sequence includes:

- a system + task setup
- an initial user query and assistant response
- a memory marker carrying forward prior context
- a follow-up user query and assistant response

Labels are masked so only assistant reply tokens contribute to loss.

Run:

```bash
python -m tracks.llm.lesson_16_compact_multi_turn_memory_sft.train --device cpu --epochs 1
```
