# Multimodal Lesson 13 Video-VLM-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_13_video_vlm_toy_temporal_qa` as an independent teaching lesson for short-video temporal QA.

## Problem

The multimodal track now covers:

- lesson 1: alignment
- lesson 2: BLIP-lite fusion
- lesson 3: LLaVA-lite visual QA
- lesson 4: grounding
- lesson 5: mask grounding
- lesson 6: Flamingo-lite interleaving
- lesson 7: Q-Former-lite bottleneck
- lesson 8: Perceiver resampling
- lesson 9: prompt-native multitask decoding
- lesson 10: open-vocabulary detection
- lesson 11: open-vocabulary segmentation
- lesson 12: key-value OCR over document images

What is still missing is a teaching-sized lesson where the visual input is a short video rather than a single image. The track currently explains multimodal reasoning over images and documents, but not over time.

## Candidate Directions

### Option A: Single-object temporal QA

- short toy video with one moving object
- ask about color, shape, or motion direction
- generate a short answer

Pros:

- cleanest introduction to temporal multimodal modeling
- keeps the time dimension as the only new challenge
- CPU-friendly

Cons:

- less visually rich than multi-object video tasks

### Option B: Multi-object temporal QA

- short toy video with multiple objects
- ask about one target object or relations
- generate a short answer

Pros:

- closer to real-world video understanding

Cons:

- adds spatial disambiguation and temporal reasoning at the same time

### Option C: Event localization

- ask when an event happens
- generate a frame index or temporal span token

Pros:

- emphasizes temporal grounding

Cons:

- less natural as a first video VLM lesson

## Recommendation

Choose Option A.

It introduces temporal reasoning while changing as little else as possible. The lesson should teach that videos are not just bigger images; they require some form of temporal aggregation before answering.

## Task Formulation

Each sample contains:

- one short video of 3-5 frames
- one moving object
- one text prompt such as:
  - `what color is the object`
  - `what shape is the object`
  - `is it moving left`
  - `is it moving up`

The model should output:

- a short answer token such as `red`, `circle`, `yes`, or `no`

## Data Design

Each video should be synthetic and CPU-friendly:

- low resolution
- one colored shape
- fixed motion direction across frames
- light background noise only

Recommended attribute family:

- colors: `red`, `green`, `blue`, `yellow`
- shapes: `square`, `circle`, `cross`
- directions: `left`, `right`, `up`, `down`

Each record should include:

- `video`
- `prompt_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `task_type`
- `prompt_text`
- `answer_text`

The prompt set should stay small and closed so the lesson focuses on temporal modeling rather than language diversity.

## Model Design

### Frame encoder

- encode each frame with the same small CNN
- produce per-frame visual tokens

### Temporal aggregator

- add a small temporal position signal
- pool or recurrently aggregate across frames
- produce a compact sequence of video tokens

### Decoder

- use a decoder-only LM like lessons 9 and 12
- prepend aggregated video tokens
- decode the answer token sequence from `video + prompt`

This keeps the lesson architecture minimal while making the new ingredient explicit:

- lesson 9 and 12: image/document prefix
- lesson 13: video prefix after temporal aggregation

## Losses and Metrics

Training loss:

- standard next-token cross entropy

Metrics:

- answer token accuracy
- answer exact match
- yes/no accuracy for direction queries

## Training Script

Follow the same conventions as earlier lessons:

- CLI args for dataset size, sequence length, image size, and text length
- `outputs/multimodal/lesson_13_video_vlm_toy_temporal_qa/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- task type
- prompt text
- answer ground truth
- answer prediction

## Success Criteria

Lesson 13 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 13
- the lesson clearly demonstrates temporal multimodal QA over short videos
