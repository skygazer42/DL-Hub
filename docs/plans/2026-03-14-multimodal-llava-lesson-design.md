# Multimodal Lesson 03 LLaVA-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_03_llava_compact_instruction_vlm` as an independent LLaVA-inspired teaching lesson for single-turn visual instruction following with short generated answers.

## Problem

The multimodal track now covers:

- lesson 1: CLIP-style alignment
- lesson 2: BLIP-lite captioning plus image-text matching

It still lacks the next step in the progression: a decoder-style instruction VLM that consumes an image plus a natural-language question and generates an answer. The user chose a short-answer, single-turn setup, so lesson 3 should teach the core LLaVA pattern without expanding into chat orchestration or long-form response generation.

## Scope

This lesson adds one full runnable module:

- `tracks/multimodal/lesson_03_llava_compact_instruction_vlm/`
- focused tests for discovery, batch contract, model outputs, and smoke training
- track README updates so the new lesson appears in the roadmap

This lesson will support only single-turn question answering with short answers. It will not implement:

- multi-turn conversation
- long descriptive answers
- external LLM backbones
- conversation templates or system prompts

## Chosen Direction

### Recommended task formulation

Use structured visual question answering with short answers:

- input: `image + instruction`
- output: one- or two-token answer

Supported instruction families:

- color questions
- shape questions
- size questions
- location questions
- yes/no questions

This gives the lesson several instruction types while keeping the answer space small and stable enough for CPU smoke training.

### Why not multi-turn chat

Multi-turn dialogue would force the lesson to carry conversation state, turn separators, and extra prompt-formatting complexity. That would dilute the main concept the lesson is supposed to teach: how visual features are projected into a decoder-style language model for conditional generation.

## Data Design

The lesson should reuse the same synthetic visual world from lessons 1 and 2:

- color
- shape
- size
- location

Each record contains:

- `image`
- `instruction_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `question_type`
- `instruction_text`
- `answer_text`

Example instruction-answer pairs:

- `what color is the shape` -> `red`
- `what shape is shown` -> `circle`
- `what size is the object` -> `small`
- `where is the object` -> `top left`
- `is the object red` -> `yes`
- `is the object at top left` -> `no`

The data format should support a decoder-only training layout:

- visual tokens are model-side prefix tokens
- text input contains BOS, instruction tokens, a separator token, and teacher-forced answer tokens
- labels ignore instruction positions and supervise only the answer span plus EOS

## Model Design

### Visual side

The visual pipeline should stay explicit and small:

- `VisionEncoder`: tiny CNN producing spatial visual tokens
- `VisionProjector`: linear projection from visual token width to language-model hidden size

The projector is a first-class teaching component, because it is the core bridge from vision features into the language model.

### Language side

The language side should be a lesson-local tiny decoder LM:

- token embedding
- recurrent or causal decoder backbone
- output vocabulary head

A small GRU-based decoder is acceptable here. It preserves causal generation behavior, keeps the code readable, and avoids the extra complexity of a full Transformer stack in a teaching lesson.

### Multimodal fusion

The key fusion mechanism is prefix conditioning:

- run projected visual tokens first
- then run instruction and answer text tokens
- produce answer logits for text positions

This gives the lesson a clear LLaVA-style story:

- encode image
- project into LM space
- condition the language model on the visual prefix
- generate answer tokens

## Losses and Metrics

Training loss:

- `qa_loss`: standard token cross entropy over answer positions only

Metrics:

- `answer_token_acc`
- `answer_exact_match`
- `yes_no_acc`

`answer_exact_match` should be the main quality metric because answers are intentionally short.

## Training Script

The training script should follow the same lesson conventions already used in the track:

- CLI args for data size, widths, epoch count, and batch caps
- `outputs/multimodal/lesson_03_llava_compact_instruction_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample rows should contain:

- instruction text
- ground-truth answer
- greedy predicted answer
- question type

## Testing Strategy

Focused tests should cover:

- `scripts/run_lesson.py` listing and dry-run resolution for lesson 3
- batch dictionaries containing instruction-VLM fields
- model outputs for logits and projected visual tokens
- finite QA loss
- smoke training run writing standard outputs

## Risks and Mitigations

- Risk: question diversity becomes too large and destabilizes smoke training.
  Mitigation: keep the instruction grammar templated and the answer space small.

- Risk: the model behaves like simple classification instead of instruction following.
  Mitigation: keep textual question tokens in the supervision path and support multiple question types over the same visual scene.

- Risk: generation code becomes harder to teach than the underlying idea.
  Mitigation: use a small causal decoder and very short answers.

## Success Criteria

Lesson 3 is complete when:

- `lesson_03_llava_compact_instruction_vlm` is discoverable through `scripts/run_lesson.py`
- it runs as a module on CPU
- it contains an independent teaching implementation
- focused tests for discovery, batch contract, model outputs, and smoke training pass
- the lesson clearly demonstrates image-conditioned instruction answering
