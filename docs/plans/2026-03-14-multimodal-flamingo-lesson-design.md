# Multimodal Lesson 06 Flamingo-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_06_flamingo_toy_interleaved_vlm` as an independent teaching lesson for interleaved image-text prompting and few-shot multimodal in-context learning.

## Problem

The multimodal track already covers:

- lesson 1: dual-encoder alignment
- lesson 2: image-conditioned generation
- lesson 3: decoder-style instruction VLM
- lesson 4: box grounding
- lesson 5: mask grounding

What is still missing is the idea that became important in Flamingo-style systems: keep a language-model-like prompt, but let images appear inside the prompt as interleaved evidence, so the model can answer a query after reading a few multimodal demonstrations.

## Candidate Directions

### Option A: Interleaved prompt with `<image>` slots

- store a text prompt containing support examples and one query
- place `<image>` tokens inside the prompt
- provide a fixed stack of support/query images aligned to those slots
- replace or inject image-aware embeddings at the `<image>` positions

Pros:

- keeps the core teaching idea visible
- easy to batch on CPU
- shows why interleaving differs from a pure visual prefix

Cons:

- simpler than true Flamingo gated cross-attention

### Option B: Separate support memory encoder

- encode support image-text pairs into a context memory
- encode the query separately
- attend over the support memory to answer the query

Pros:

- simple implementation
- explicit few-shot memory

Cons:

- hides the interleaved prompt idea
- feels more like retrieval or meta-learning than Flamingo

### Option C: Mini gated cross-attention decoder

- keep a causal text stream
- add gated cross-attention blocks that consult resampled image latents

Pros:

- closest to paper intuition

Cons:

- too much implementation noise for a teaching-sized CPU lesson

## Recommendation

Choose Option A.

It keeps the lesson strongly connected to Flamingo's central teaching idea, while staying small enough for a local lesson. The implementation can honestly be described as "Flamingo-lite" rather than a faithful paper reproduction.

## Task Formulation

Each training example contains:

- two support image-text demonstrations
- one query image-text prompt
- one short target answer

The prompt is interleaved:

- `example <image> what is dax <sep> red`
- `example <image> what is dax <sep> blue`
- `query <image> what is dax <sep>`

The answer tokens for the query are supervised after the final separator.

## Why Few-Shot Context Matters

The synthetic prompt token such as `dax` should not have a fixed meaning across the whole dataset.

Instead, for each sample:

- choose one hidden attribute family: color, shape, size, or location
- choose one synthetic task word such as `dax`, `blicket`, `wug`, or `zup`
- support examples reveal what the task word means through their answers
- the query uses the same task word on a new image

This makes the support context necessary. Without the demonstrations, the query word is ambiguous.

## Data Design

Each sample should include:

- `images`: `(num_images, 3, H, W)` where `num_images = num_shots + 1`
- `prompt_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `task_token`
- `attribute_name`
- `answer_text`

Recommended defaults:

- `num_shots = 2`
- `image_size = 16`
- `max_text_length = 28`

The same simple colored-shape renderer used in earlier lessons is good enough. The key change is the prompt structure, not the image complexity.

## Model Design

### Vision side

- encode each image independently with a tiny CNN
- pool each image into one compact embedding

### Text side

- embed the prompt tokens
- identify `<image>` token positions
- inject the aligned image embeddings at those positions

### Sequence model

- run a small decoder-like GRU over the interleaved sequence
- produce logits at every token position
- compute loss only on the query answer suffix

This is intentionally simpler than a real Transformer with gated cross-attention, but it cleanly teaches:

- why image positions matter inside a prompt
- how support examples and query examples can share one multimodal sequence

## Losses and Metrics

Training loss:

- token cross entropy on the query answer tokens only

Metrics:

- answer token accuracy
- answer exact match
- context usage accuracy on samples where support examples are informative

The third metric can be approximated by filtering to ordinary evaluation samples and checking exact answer correctness. It is acceptable to log only token accuracy and exact match in the first delivery if the code stays clearer.

## Training Script

Follow the same conventions as lessons 1-5:

- CLI args for dataset size and text length
- `outputs/multimodal/lesson_06_flamingo_toy_interleaved_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- task token
- hidden attribute family
- prompt text
- ground-truth answer
- predicted answer

## Success Criteria

Lesson 6 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the README progression includes lesson 6
- the lesson clearly demonstrates interleaved few-shot multimodal prompting
