# Lesson 23: Toy Critic Rerank Prompting

This lesson demonstrates a tiny critic-rerank prompting template on synthetic sequences.

Each sample contains:

- a `PROMPT` token and topic token
- two `CANDIDATE` spans, each with short candidate tokens
- a critic `SCORE` token after each candidate span
- a `RERANK` marker followed by the selected best answer tokens

Training uses response-only masking for the reranked answer span: candidate and score traces are context only, while reranked answer tokens plus EOS are supervised.

Run:

```bash
python -m tracks.llm.lesson_23_toy_critic_rerank_prompting.train --device cpu --epochs 1
```
