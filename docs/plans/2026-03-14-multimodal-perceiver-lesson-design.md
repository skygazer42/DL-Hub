# Multimodal Lesson 08 Perceiver-Resampler-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_08_perceiver_resampler_toy_vlm` as an independent teaching lesson for fixed-latent visual resampling in the style of Perceiver Resampler systems.

## Problem

The multimodal track already covers:

- lesson 1: dual-encoder alignment
- lesson 2: BLIP-lite fusion
- lesson 3: LLaVA-lite visual prefixing
- lesson 4: box grounding
- lesson 5: mask grounding
- lesson 6: Flamingo-lite interleaved prompting
- lesson 7: Q-Former-lite query bottleneck

What is still missing is the pattern where a model receives a large collection of visual tokens, often from multiple views or many patches, then compresses them into a small fixed set of latent tokens before decoding. That is the teaching role for a Perceiver-style resampler.

## Candidate Directions

### Option A: Single-image dense token compression

- take one image
- produce many spatial tokens
- compress them to fixed latents
- decode an answer

Pros:

- simple

Cons:

- not obviously different from lesson 7

### Option B: Multi-view scene with fixed-latent resampling

- render one scene
- provide a full-scene view plus quadrant crops
- flatten all visual tokens from all views
- resample to a compact latent array
- answer a question about one object in the scene

Pros:

- makes the resampling motivation obvious
- teaches why fixed latent counts help with many visual tokens
- stays CPU-friendly

Cons:

- dataset is slightly larger than single-view QA

### Option C: Video-frame resampler

- provide a short sequence of frames
- resample temporal-spatial tokens
- answer a question about motion or state

Pros:

- realistic downstream motivation

Cons:

- too much complexity for the current teaching step

## Recommendation

Choose Option B.

It cleanly separates lesson 8 from lesson 7. Lesson 7 already uses learned queries on one image; lesson 8 should show a many-token, multi-view input collapsed to a stable latent budget.

## Task Formulation

Each sample contains five images:

- one full-scene image
- four quadrant crops, resized back to the lesson image size

The scene contains one object per quadrant. The question asks about a specific quadrant, for example:

- `what color is the object at top left`
- `what shape is the object at bottom right`
- `what size is the object at top left`
- `is the object at bottom left red`

This makes the crop views useful while keeping the answer short and clean.

## Data Design

Each record should include:

- `images`: `(num_views, 3, H, W)` with `num_views = 5`
- `question_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `question_type`
- `question_text`
- `answer_text`

Recommended defaults:

- `image_size = 16`
- `scene_size = 32`
- `max_text_length = 14`

The supervision target should only score the answer tokens, not the question prefix.

## Model Design

### Vision encoder

- encode each view independently with a tiny CNN
- keep spatial tokens from each view
- flatten tokens across views into one large token set

### Perceiver resampler

- start from a fixed set of learned latent tokens
- run latent self-attention
- run cross-attention from latents to visual tokens
- update latents with a small feed-forward block

Use one or two blocks only. The goal is to teach the mechanism, not chase scale.

### Decoder LM

- project or directly consume the resampled latent tokens
- prepend them to a tiny decoder-style GRU language model
- decode the question-plus-answer sequence

This creates a clear contrast with lesson 7:

- lesson 7: learned queries read a single image token set
- lesson 8: learned latents resample a larger multi-view token pool

## Losses and Metrics

Training loss:

- token cross entropy over answer tokens only

Metrics:

- answer token accuracy
- answer exact match
- yes/no accuracy

## Training Script

Follow the same conventions as lessons 1-7:

- CLI args for sample count, image size, scene size, text length, and latent count
- `outputs/multimodal/lesson_08_perceiver_resampler_toy_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- question type
- question text
- answer gt
- answer pred

## Success Criteria

Lesson 8 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 8
- the lesson clearly demonstrates multi-view token resampling into fixed latents
