# Lesson 04: Toy Instruction Tuning

This lesson turns the tiny causal LM from lesson 01 into a single-turn instruction-following toy.

What changes relative to chat SFT:

- the prompt is structured as `instruction -> input -> response`
- only response tokens contribute to the loss
- the target response is a deterministic transform of the prompt content, so the full loop stays local and CPU-friendly

Run:

```bash
python -m tracks.llm.lesson_04_toy_instruction_tuning.train --device cpu --epochs 1
```
