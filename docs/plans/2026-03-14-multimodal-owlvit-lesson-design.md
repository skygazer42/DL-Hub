# Multimodal Lesson 10 OWL-ViT-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection` as an independent teaching lesson for open-vocabulary detection with text queries.

## Problem

The multimodal track already covers:

- lesson 1: alignment
- lesson 2: BLIP-lite fusion
- lesson 3: LLaVA-lite visual QA
- lesson 4: grounding
- lesson 5: mask grounding
- lesson 6: Flamingo-lite interleaving
- lesson 7: Q-Former-lite bottleneck
- lesson 8: Perceiver resampling
- lesson 9: PaliGemma-lite prompt-native multitask decoding

What is still missing is the open-vocabulary detection setting where the detector is conditioned by text queries rather than a fixed classifier head. That is the teaching role for an OWL-ViT-like lesson.

## Candidate Directions

### Option A: Query-conditioned positive-only localization

- each sample contains one image and one query
- query always matches an object
- predict the object box

Pros:

- simple

Cons:

- too close to lesson 4
- misses the open-vocabulary detection idea that queries may be absent

### Option B: Query-conditioned presence plus localization

- image contains multiple objects
- query may match or may be absent
- predict presence and, if present, the object location

Pros:

- captures the core open-vocabulary detection behavior
- stays small and CPU-friendly
- clearly differs from lesson 4

Cons:

- needs masked regression on positives only

### Option C: Multi-query set prediction

- one image, several candidate queries
- predict a score and box for every query

Pros:

- closest to practical detectors

Cons:

- too much batching complexity for this lesson

## Recommendation

Choose Option B.

It keeps the lesson small while still teaching the core open-vocabulary detection pattern: text query in, presence + localization out.

## Task Formulation

Each sample contains:

- one image with 2-4 colored shapes
- one text query such as `detect red square`

The query is positive for some samples and negative for others:

- positive: the queried category exists in the image
- negative: the queried category is absent

The model should output:

- `present / absent`
- a grid cell and box when present

## Data Design

Each record should include:

- `image`
- `query_ids`
- `attention_mask`
- `target_present`
- `target_cell`
- `target_delta`
- `target_box`
- `query_text`

Recommended defaults:

- `image_size = 32`
- `grid_size = 4`
- `max_text_length = 6`

The data generator should ensure unique color-shape pairs inside one image so that the query is unambiguous.

## Model Design

### Vision side

- tiny CNN over the image
- produce one feature per grid cell

### Text side

- embedding plus masked mean pooling
- project into the same hidden space

### Fusion and heads

- broadcast text embedding over the spatial map
- fuse text and visual features per cell
- predict:
  - presence logit from pooled fused features
  - cell logits from the spatial map
  - per-cell box deltas

This keeps the architecture parallel to lesson 4 while changing the task framing:

- lesson 4: positive referring expression grounding
- lesson 10: open-vocabulary query detection with possible absence

## Losses and Metrics

Training loss:

- BCE for presence
- cell cross entropy for positives only
- box regression for positives only

Recommended total:

- `presence_loss + cell_loss + box_weight * box_loss`

Metrics:

- presence accuracy
- positive-only bbox L1
- positive-only center accuracy

## Training Script

Follow the same conventions as lessons 1-9:

- CLI args for dataset size, image size, grid size, text length, and batch caps
- `outputs/multimodal/lesson_10_owlvit_compact_open_vocab_detection/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- query text
- target present
- predicted present
- target box
- predicted box

## Success Criteria

Lesson 10 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 10
- the lesson clearly demonstrates text-conditioned open-vocabulary detection
