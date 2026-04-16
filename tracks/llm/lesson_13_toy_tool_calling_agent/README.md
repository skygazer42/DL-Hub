# Lesson 13: Toy Tool-Calling Agent

This lesson implements a tiny synthetic tool-calling loop for a language model.

Each sample contains:

- a user prompt with two synthetic number tokens
- an explicit tool-call step (`calculator` or `lookup`)
- a result token generated from the selected tool

The model predicts:

- next-token logits for the response sequence
- a tool-selection head (`num_tools=2`)

Run:

```bash
python -m tracks.llm.lesson_13_toy_tool_calling_agent.train --device cpu --epochs 1
```
