# Multimodal Lesson 14 BMN-Lite Temporal Grounding Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_14_bmn_compact_temporal_grounding` as an independent teaching lesson for text-conditioned temporal localization over short videos.

## Problem

The multimodal track now covers image-text alignment, image-conditioned generation, grounding, open-vocabulary detection and segmentation, document OCR, and short-video QA.

What is still missing is a lesson that teaches temporal localization itself:

- a video is not only something to classify or answer questions about
- a text query can refer to a span in time
- temporal understanding often needs boundary prediction plus proposal scoring

The next lesson should introduce temporal grounding without adding too many new moving parts.

## Candidate Directions

### Option A: Start-End Boundary Prediction Only

- encode video and query
- predict one start index and one end index

Pros:

- smallest implementation
- very easy to explain

Cons:

- does not really teach boundary-matching
- too far from the BMN family the user requested

### Option B: BMN-Lite Temporal Grounding

- encode video and query
- predict start boundary scores
- predict end boundary scores
- build an upper-triangular proposal score map for all valid segments

Pros:

- directly matches the requested Boundary / Boundary-Matching direction
- teaches the difference between boundary supervision and proposal supervision
- naturally supports temporal IoU metrics

Cons:

- more code than direct start-end regression

### Option C: Anchor-Segment Temporal Detection

- define temporal anchors
- classify and regress anchors against one ground-truth segment

Pros:

- conceptually similar to object detection lessons

Cons:

- anchor-based formulation distracts from the requested BMN-style idea

## Recommendation

Choose Option B.

This lesson should be a teaching-first BMN-lite formulation:

- one query
- one video
- one target segment
- one compact proposal map

That keeps the core idea visible:

- boundary scores say where an event may begin and end
- boundary-matching scores say how good each candidate segment is

## Task Formulation

Each sample contains:

- one short video with one colored shape
- one text query that describes a temporal event
- one ground-truth segment `(start_idx, end_idx)` inside the clip

Example query forms:

- `when does the red square move left`
- `when does the blue circle move right`
- `when does the green cross flash`

The model should predict:

- start boundary confidence over all frames
- end boundary confidence over all frames
- a proposal confidence map over all valid `(start, end)` pairs
- one final best segment decoded from those scores

## Data Design

Each video should be synthetic and CPU-friendly:

- low resolution
- one object only
- fixed short clip length
- light background noise

Recommended event families:

- `move left`
- `move right`
- `flash`

For motion events:

- the object stays mostly stable outside the target segment
- inside the target segment it moves clearly in the requested direction

For flash events:

- the object keeps its position
- inside the target segment its brightness or color changes sharply

Each record should include:

- `video`
- `query_ids`
- `attention_mask`
- `start_labels`
- `end_labels`
- `proposal_labels`
- `proposal_mask`
- `segment`
- `query_text`
- `event_type`

The proposal supervision should use temporal IoU between every valid proposal and the ground-truth segment. That makes the upper-triangular proposal map interpretable and keeps the loss definition simple.

## Model Design

### Video encoder

- shared small CNN on every frame
- temporal encoder over frame features
- output sequence shape `(B, T, H)`

### Text encoder

- small token embedding layer
- one lightweight GRU
- pooled query embedding `(B, H)`

### Fusion

- broadcast the query embedding across time
- fuse query and temporal video features with a small MLP
- produce text-conditioned temporal features `(B, T, H)`

### Heads

- `start_head`: per-frame boundary start logit
- `end_head`: per-frame boundary end logit
- `proposal_head`: upper-triangular score map over all valid `(start, end)` proposals

The proposal feature should combine:

- start feature
- end feature
- pooled feature over the segment interior

This is the minimal teaching version of boundary-matching.

## Losses and Metrics

Training losses:

- start boundary BCE
- end boundary BCE
- proposal regression loss against temporal IoU targets

Recommended total loss:

- `start_loss + end_loss + proposal_loss`

Metrics:

- start accuracy
- end accuracy
- mean temporal IoU between predicted best segment and target
- `R@1, IoU=0.5`

## Training Script

Follow the same conventions as earlier lessons:

- CLI args for dataset size, frame count, image size, and query length
- `outputs/multimodal/lesson_14_bmn_compact_temporal_grounding/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- query text
- event type
- target segment
- predicted segment
- temporal IoU

## Success Criteria

Lesson 14 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- data tests pass
- model and loss tests pass
- CPU smoke training passes
- the multimodal track README includes lesson 14
- the lesson clearly demonstrates boundary prediction plus boundary-matching temporal grounding
