# Multimodal Lesson 16 Multi-Scale 2D-TAN-Lite Temporal Grounding Design

**Date:** 2026-03-15

**Goal:** Add `tracks/multimodal/lesson_16_multiscale_2dtan_toy_temporal_grounding` as the next teaching lesson after lesson 15, extending dense temporal grounding from a single-scale `T x T` segment map to a multi-scale temporal-map formulation.

## Problem

The multimodal track already includes two temporal grounding lessons:

- lesson 14 teaches BMN-lite boundary prediction plus proposal matching
- lesson 15 teaches direct dense `T x T` segment-map reasoning with a single temporal scale

What is still missing is the next natural temporal-localization idea:

- keeping the same grounding task
- keeping the same single-query and single-target lesson scope
- but showing why temporal grounding benefits from coarse and fine temporal resolutions at the same time

The next lesson should therefore stay close to lesson 15 in task definition while making one clearly visible architectural change:

- single-scale temporal map becomes multi-scale temporal maps plus fusion

## Candidate Directions

### Option A: Query-Refined 2D-TAN-Lite

- keep one temporal scale
- apply several rounds of query-conditioned refinement on the same map

Pros:

- implementation is small
- training remains simple

Cons:

- too similar to lesson 15
- does not clearly teach why multi-scale temporal structure matters

### Option B: Multi-Scale 2D-TAN-Lite

- build temporal features at multiple temporal resolutions
- construct one segment map per scale
- fuse them back into a final fine-resolution prediction

Pros:

- strongest conceptual continuation from lesson 15
- teaches coarse-to-fine temporal reasoning
- keeps the task fixed while introducing one meaningful new design idea

Cons:

- needs scale-specific supervision and map fusion logic

### Option C: Multi-Moment Retrieval-Lite

- predict and rank multiple temporal moments instead of one best segment

Pros:

- closer to retrieval-style temporal grounding settings

Cons:

- changes labels, decoding, and metrics all at once
- too large a jump for the immediate next lesson

## Recommendation

Choose Option B.

Lesson 16 should be a teaching-first multi-scale extension of lesson 15:

- one query
- one short video
- one target moment
- one final fused fine-resolution segment map
- three supervised scales so students can compare coarse and fine predictions directly

That produces a clean progression:

- lesson 14: boundary-first grounding
- lesson 15: single-scale dense segment-map grounding
- lesson 16: multi-scale dense segment-map grounding

## Task Formulation

Each sample contains:

- one short video with one colored shape
- one text query that refers to an event span
- one ground-truth segment `(start_idx, end_idx)`

The task scope remains deliberately narrow:

- single query
- single object
- single event span
- one best predicted segment at inference time

This avoids mixing a new architectural idea with a new task definition.

## Data Design

The lesson should remain CPU-friendly and close to lessons 14 and 15:

- one object only
- short clip length
- low spatial resolution
- fixed prompt templates
- one target segment per query

Recommended event families remain:

- `move left`
- `move right`
- `flash`

Each record should include the lesson 15 fields:

- `video`
- `query_ids`
- `attention_mask`
- `segment`
- `query_text`
- `event_type`

In addition, the supervision expands from one dense map to three scales:

- `map_labels_s1`
- `map_mask_s1`
- `map_labels_s2`
- `map_mask_s2`
- `map_labels_s3`
- `map_mask_s3`

The scale definitions should be:

- scale 1: original temporal resolution `T`
- scale 2: pooled temporal resolution `T / 2`
- scale 3: pooled temporal resolution `T / 4`

Each scale still uses an upper-triangular valid-cell mask and temporal-IoU soft labels. The only difference is that each coarse cell represents a coarser temporal interval. This keeps the lesson interpretable:

- every valid cell is still a candidate segment
- every scale still predicts overlap with the target moment
- students can compare how localization changes across resolutions

To keep implementation simple and stable, the default frame count should remain divisible by 4.

## Model Design

### Shared encoders

The lesson should preserve the same front-end pattern as lesson 15:

- a small shared CNN encodes each video frame
- a lightweight temporal encoder produces frame features `(B, T, H)`
- a lightweight query GRU produces one query embedding `(B, H)`

This keeps the new lesson focused on temporal-scale structure rather than on changing the backbone.

### Multi-scale temporal features

After temporal encoding, build a short temporal pyramid:

- scale 1 uses the original sequence
- scale 2 pools adjacent temporal steps
- scale 3 pools again to an even coarser sequence

Average pooling is sufficient for the teaching version because the lesson is about resolution changes, not about sophisticated temporal aggregation.

### Per-scale segment maps

At each scale, build a dense upper-triangular temporal map using the same cell recipe as lesson 15:

- start feature
- end feature
- pooled interior segment feature
- query embedding

Project each valid `(start, end)` cell into a segment feature tensor and refine it with a small 2D convolutional stack. Predict one score map per scale:

- `score_map_s1`
- `score_map_s2`
- `score_map_s3`

### Multi-scale fusion

The coarse score maps should be resized back to the fine `T x T` resolution and fused with the fine-scale map. The teaching version should keep fusion explicit and simple:

- upsample `score_map_s2` to `T x T`
- upsample `score_map_s3` to `T x T`
- concatenate or sum the aligned maps
- apply a small fusion head to produce `fused_score_map`

Inference should decode one best segment from `fused_score_map` only. This preserves a direct comparison with lesson 15.

## Losses and Metrics

Training should use deep supervision across scales:

- main fused loss on `fused_score_map`
- auxiliary losses on `score_map_s1`, `score_map_s2`, and `score_map_s3`

The recommended total loss is:

`loss = fused_loss + 0.5 * mean(aux_s1, aux_s2, aux_s3)`

Each individual loss should remain a masked MSE between sigmoid-normalized score maps and temporal-IoU labels.

Metrics should stay identical to lesson 15 so comparisons stay easy:

- mean temporal IoU
- `R@1, IoU=0.5`
- `R@1, IoU=0.7`

Sample logging should include:

- query text
- event type
- target segment
- `pred_segment_s1`
- `pred_segment_s2`
- `pred_segment_s3`
- `pred_segment_fused`
- temporal IoU for the fused prediction

## Lesson Positioning

The README should make the contrast with lesson 15 explicit:

- lesson 15: one temporal resolution, one segment map
- lesson 16: several temporal resolutions, several segment maps, one fused prediction

The main new teaching ideas should be:

- coarse temporal scales carry broader temporal context
- fine temporal scales preserve boundary precision
- deep supervision can stabilize learning across resolutions

## Training Script

Follow the existing track conventions:

- CLI args for dataset size, frame count, image size, and query length
- output directory under `outputs/multimodal/lesson_16_multiscale_2dtan_toy_temporal_grounding/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

The lesson should remain runnable on CPU with a short smoke command.

## Success Criteria

Lesson 16 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- data tests pass with all three supervision scales
- model and loss tests pass
- CPU smoke training passes
- the multimodal track README includes lesson 16
- the lesson README clearly explains how multi-scale temporal maps extend lesson 15
- sample outputs make per-scale predictions inspectable
