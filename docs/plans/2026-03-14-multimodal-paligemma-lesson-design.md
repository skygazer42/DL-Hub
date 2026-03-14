# Multimodal Lesson 09 PaliGemma-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_09_paligemma_toy_siglip_decoder_vlm` as an independent teaching lesson for prompt-native multitask text generation with a SigLIP-style vision tower and decoder-only language model.

## Problem

The multimodal track already covers:

- lesson 1: alignment
- lesson 2: BLIP-lite fusion
- lesson 3: LLaVA-lite visual instruction VLM
- lesson 4: box grounding
- lesson 5: mask grounding
- lesson 6: Flamingo-lite interleaving
- lesson 7: Q-Former-lite bottleneck
- lesson 8: Perceiver-resampler-lite multi-view compression

What is still missing is the idea that a single decoder-only interface can handle multiple vision tasks by expressing every target as text. That is the teaching role for a PaliGemma-like lesson.

## Candidate Directions

### Option A: Pure visual QA with a different vision encoder

- keep one QA task
- swap in a stronger image encoder

Pros:

- simple

Cons:

- too close to lesson 3
- does not highlight the unified text output interface

### Option B: Prompt-native multitask text generation

- use one image
- switch between prompts such as captioning, attribute QA, location output, and yes/no
- always predict text

Pros:

- clearly shows the “everything is text” idea
- keeps data small and CPU-friendly
- cleanly fits a PaliGemma-like decoder interface

Cons:

- less architecturally novel than lessons 7 and 8

### Option C: OCR-like reading lesson

- render short text in the image
- ask the model to transcribe or answer OCR questions

Pros:

- realistic downstream use case

Cons:

- more rendering and vocabulary complexity than needed here

## Recommendation

Choose Option B.

Lesson 9 should teach task unification rather than yet another bottleneck mechanism. The model can stay modest while the prompt/output interface becomes more general.

## Task Formulation

Each sample contains one synthetic image with one object. The prompt selects a task:

- `caption the image`
- `answer color`
- `answer shape`
- `locate object`
- `is object red`

The target is always text:

- `red square top left`
- `red`
- `square`
- `top left`
- `yes`

This mirrors the key lesson: one decoder, many tasks, one text interface.

## Data Design

Each record should include:

- `image`
- `prompt_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `task_type`
- `prompt_text`
- `answer_text`

Recommended defaults:

- `image_size = 16`
- `max_text_length = 16`

The object can vary in:

- color
- shape
- size
- location

The supervision target should only score the answer tokens, not the prompt prefix.

## Model Design

### Vision side

Use a small SigLIP-style teaching tower:

- CNN patch features
- patch-token output
- linear projector into decoder hidden space

The point is not to reproduce SigLIP training, but to teach the pattern of a compact vision tower feeding a decoder-only LM.

### Decoder side

- embed prompt tokens
- prepend or inject projected visual tokens
- decode prompt-plus-answer sequence with a small GRU LM

This keeps the implementation transparent and CPU-runnable.

## Losses and Metrics

Training loss:

- token cross entropy over answer tokens only

Metrics:

- answer token accuracy
- answer exact match
- yes/no accuracy
- location exact match for `locate` prompts

It is acceptable for the first version to log token accuracy, exact match, and yes/no accuracy only.

## Training Script

Follow the same conventions as lessons 1-8:

- CLI args for sample count, image size, and text length
- `outputs/multimodal/lesson_09_paligemma_toy_siglip_decoder_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- task type
- prompt text
- answer gt
- answer pred

## Success Criteria

Lesson 9 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 9
- the lesson clearly demonstrates prompt-native multitask text output
