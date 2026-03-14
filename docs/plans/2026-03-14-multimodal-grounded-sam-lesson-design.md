# Multimodal Lesson 11 Grounded-SAM-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation` as an independent teaching lesson for open-vocabulary text-conditioned segmentation with possible query absence.

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
- lesson 9: PaliGemma-lite prompt-native multitask decoding
- lesson 10: OWL-ViT-lite open-vocabulary detection

What is still missing is the open-vocabulary segmentation setting where a text query may or may not match an object, and the model must decide both whether the object exists and what region belongs to it. That is the teaching role for a Grounded-SAM-like lesson.

## Candidate Directions

### Option A: Positive-only text-to-mask grounding

- image contains multiple objects
- query always matches one object
- predict only the target mask

Pros:

- simple

Cons:

- too close to lesson 5
- misses the open-vocabulary absence case learned in lesson 10

### Option B: Text-conditioned presence plus mask prediction

- image contains multiple objects
- query may match or may be absent
- predict presence and, if present, the target mask

Pros:

- clean combination of lesson 5 and lesson 10
- teaches open-vocabulary segmentation directly
- remains CPU-friendly with low-resolution masks

Cons:

- needs masked segmentation loss on positives only

### Option C: Text plus point prompt segmentation

- query text specifies the category
- an extra point prompt indicates the target instance
- output the mask

Pros:

- feels closer to interactive SAM

Cons:

- adds prompt complexity that distracts from the open-vocabulary concept

## Recommendation

Choose Option B.

It is the clearest next lesson after lesson 10 because it upgrades open-vocabulary detection into open-vocabulary segmentation without introducing extra prompt modalities.

## Task Formulation

Each sample contains:

- one image with 2-4 colored shapes
- one text query such as `segment red square` or `mask blue circle`

The query is positive for some samples and negative for others:

- positive: the queried category exists in the image
- negative: the queried category is absent

The model should output:

- `present / absent`
- a low-resolution foreground mask when present

## Data Design

Each record should include:

- `image`
- `query_ids`
- `attention_mask`
- `target_present`
- `target_mask`
- `query_text`

Recommended defaults:

- `image_size = 32`
- `mask_size = 8`
- `max_text_length = 6`

The data generator should ensure unique color-shape pairs inside one image so each positive query is unambiguous. Negative queries should be sampled from absent color-shape pairs. The low-resolution supervision mask should be produced by downsampling the matched object's binary mask.

## Model Design

### Image encoder

- tiny CNN over the image
- output a low-resolution spatial feature map aligned to `mask_size`

### Text prompt encoder

- token embedding plus masked mean pooling
- project the text query into a prompt embedding

### Grounded-SAM-lite decoder

Use a prompt-encoder plus mask-decoder style formulation:

- broadcast the prompt embedding over the visual map
- fuse visual and prompt features
- predict `mask_logits` from the fused map
- predict `presence_logit` from pooled fused features

This keeps the lesson small while still making the architecture read as:

- prompt encoder
- mask decoder
- presence head

## Losses and Metrics

Training loss:

- BCE for presence
- BCE for mask on positives only
- dice loss for mask on positives only

Recommended total:

- `presence_loss + mask_bce_loss + dice_weight * mask_dice_loss`

Metrics:

- presence accuracy
- positive-only mask IoU
- positive-only dice score
- positive-only foreground accuracy

## Training Script

Follow the same conventions as earlier lessons:

- CLI args for dataset size, image size, mask size, text length, and batch caps
- `outputs/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- query text
- target present
- predicted present
- target foreground ratio
- predicted foreground ratio

## Success Criteria

Lesson 11 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 11
- the lesson clearly demonstrates open-vocabulary text-conditioned segmentation with possible absence
