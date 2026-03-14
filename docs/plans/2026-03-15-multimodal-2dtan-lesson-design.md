# Multimodal Lesson 15 2D-TAN-Lite Temporal Grounding Design

**Date:** 2026-03-15

**Goal:** Add `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding` as an independent teaching lesson for text-conditioned temporal localization with a dense 2D temporal map.

## Problem

The multimodal track now covers:

- image-text alignment and generation
- grounding and open-vocabulary segmentation
- document OCR
- short-video QA
- BMN-lite temporal grounding

What is still missing is a lesson that teaches a second major temporal grounding formulation:

- not boundary-first
- not proposal matching as a separate head
- but direct reasoning over a dense 2D segment map

The next lesson should keep the same single-query temporal grounding task while changing the model structure enough that students can see a clearly different design pattern from lesson 14.

## Candidate Directions

### Option A: VSLNet-Lite

- keep start/end localization
- add query-guided temporal interaction

Pros:

- small implementation delta from lesson 14
- easy to train

Cons:

- too similar to lesson 14
- does not make the 2D segment-map idea explicit

### Option B: 2D-TAN-Lite

- encode video and query
- lift 1D temporal features into a `T x T` segment map
- score all valid `(start, end)` cells directly

Pros:

- very clear contrast with lesson 14
- teaches dense segment-map reasoning directly
- keeps the temporal grounding task unchanged while changing the model bias

Cons:

- needs an extra 2D map construction step

### Option C: Moment-DETR-Lite

- use query-conditioned set prediction for moments

Pros:

- modern architecture flavor

Cons:

- too large a conceptual jump for the next lesson
- adds matching and set prediction complexity too early

## Recommendation

Choose Option B.

Lesson 15 should be a teaching-first 2D-TAN-lite formulation:

- one query
- one short video
- one target moment
- one dense upper-triangular temporal map

That makes the lesson contrast simple and memorable:

- lesson 14: boundary prediction plus proposal matching
- lesson 15: direct dense scoring over a 2D segment map

## Task Formulation

Each sample contains:

- one short video with one colored shape
- one text query that refers to an event span
- one ground-truth segment `(start_idx, end_idx)`

Example query forms:

- `when does the red square move left`
- `when does the blue circle move right`
- `when does the green cross flash`

The model should predict:

- one dense `T x T` score map over valid `(start, end)` segments
- one final best segment decoded from that map

## Data Design

The lesson should stay CPU-friendly and close to lesson 14:

- one object only
- short clip length
- low spatial resolution
- fixed prompt templates
- one target segment per query

Recommended event families:

- `move left`
- `move right`
- `flash`

Each record should include:

- `video`
- `query_ids`
- `attention_mask`
- `map_labels`
- `map_mask`
- `segment`
- `query_text`
- `event_type`

The dense map supervision should use temporal IoU between every valid `(start, end)` proposal and the target segment. That makes the map interpretable:

- each cell is a candidate segment
- larger values mean higher overlap with the ground truth

The upper-triangular structure should stay explicit through `map_mask`.

## Model Design

### Video encoder

- shared small CNN on every frame
- lightweight temporal encoder over frame features
- output 1D sequence shape `(B, T, H)`

### Text encoder

- small token embedding layer
- one lightweight GRU
- pooled query embedding `(B, H)`

### 2D temporal map construction

For each valid `(start, end)` pair:

- take the start feature
- take the end feature
- take the pooled feature over the segment interior
- concatenate with the query embedding
- project to one segment-level cell feature

This produces a dense map with shape `(B, T, T, H)`.

### 2D map reasoning

- apply a small 2D convolutional stack over the temporal map
- predict one scalar score per cell
- use only the upper triangle as valid output

This is the minimal teaching version of 2D-TAN-style dense temporal reasoning.

## Losses and Metrics

Training loss:

- masked MSE between predicted `score_map` and `map_labels`

Metrics:

- mean temporal IoU between the best predicted segment and target
- `R@1, IoU=0.5`
- optional `R@1, IoU=0.7`

## Training Script

Follow the same conventions as earlier lessons:

- CLI args for dataset size, frame count, image size, and query length
- `outputs/multimodal/lesson_15_2dtan_toy_temporal_grounding/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- query text
- event type
- target segment
- predicted segment
- temporal IoU

## Success Criteria

Lesson 15 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- data tests pass
- model and loss tests pass
- CPU smoke training passes
- the multimodal track README includes lesson 15
- the lesson clearly demonstrates dense 2D temporal-map reasoning for text-conditioned temporal grounding
