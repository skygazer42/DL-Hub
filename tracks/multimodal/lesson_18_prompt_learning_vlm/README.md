# Lesson 18: Prompt Learning for a Compact VLM

This lesson demonstrates a tiny CoOp-style adaptation setup for multimodal retrieval.

- A small image encoder and text encoder are initialized once and frozen.
- The only trainable adaptation path is a bank of soft prompt tokens prepended to each text concept.
- Synthetic color-and-shape images keep the task CPU friendly while making the prompt mechanism explicit.

Run:

```bash
python -m tracks.multimodal.lesson_18_prompt_learning_vlm.train --device cpu --epochs 3
```
