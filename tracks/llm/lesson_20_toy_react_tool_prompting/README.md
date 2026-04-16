# Lesson 20: Toy ReAct Tool Prompting

This lesson demonstrates a tiny ReAct-style prompting template on synthetic sequences.

Each sample contains:

- a `REACT` prompt token and topic token
- a `THINK` token with a short internal thought token
- an `ACT` token with a synthetic tool-choice token
- an `OBSERVE` token with a synthetic observation token
- a `FINAL` marker followed by the final answer tokens

Training uses response-only masking for the final answer span: pre-final reasoning and tool steps are context only, while final answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_20_toy_react_tool_prompting.train --device cpu --epochs 1
```
