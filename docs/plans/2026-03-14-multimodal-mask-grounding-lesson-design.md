# Multimodal Lesson 05 Mask-Grounding-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_05_mask_grounding_toy_refexp` as an independent teaching lesson for text-conditioned region grounding with low-resolution target masks.

## Problem

The multimodal track now covers:

- lesson 1: CLIP-style alignment
- lesson 2: BLIP-lite captioning plus ITM
- lesson 3: LLaVA-lite instruction answering
- lesson 4: grounding-lite bbox localization

What is still missing is the natural next step after bbox grounding: region grounding. The user chose mask grounding, and the teaching-friendly version is a low-resolution segmentation objective that shows how language can guide per-location region prediction without the complexity of a full-resolution segmentation framework.

## Scope

This lesson adds one full runnable module:

- `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/`
- focused tests for discovery, batch contract, model outputs, and smoke training
- an update to `tracks/multimodal/README.md` so lesson 5 appears in the progression

This lesson will support:

- single-target referring-expression mask grounding
- low-resolution target masks
- text-conditioned spatial fusion

It will not support:

- full-resolution masks
- multi-instance masks
- interactive segmentation prompts
- mask refinement stages

## Chosen Direction

### Recommended formulation

Use low-resolution mask prediction:

- input: `image + referring expression`
- output: a low-resolution target mask

This keeps the lesson stable and CPU-friendly while still teaching the key concept: language-conditioned dense spatial prediction.

### Why not full-resolution masks

Full-resolution segmentation would add more compute and more implementation noise than teaching value. The low-resolution version is sufficient to explain:

- per-location conditioning
- foreground vs background prediction
- dense region grounding rather than bbox grounding

## Data Design

The lesson should reuse the same multi-object synthetic scene style as lesson 4:

- multiple colored shapes
- one unique textual target

Each record contains:

- `image`
- `input_ids`
- `attention_mask`
- `target_mask`
- `query_text`

The query can be phrased with templates such as:

- `segment the red square`
- `mask the small blue circle`
- `highlight the green object at top left`

The target should be a binary low-resolution mask, for example `8 x 8`, produced by downsampling the target object's full-resolution binary mask.

## Model Design

### Vision side

Use a tiny CNN that preserves a small spatial map:

- convolution stack
- adaptive pooling to the low-resolution mask size

### Text side

Use a lightweight text encoder:

- token embedding
- masked mean pooling

### Fusion

Broadcast the text embedding across the spatial feature map and fuse it per location. This keeps the lesson parallel to lesson 4 while changing the output head from box prediction to dense mask prediction.

### Mask head

The head should output:

- `mask_logits`: `(B, 1, Hm, Wm)`
- `pred_mask`: probabilities or thresholded predictions at the same low resolution

Optional upsampling can be used only for visualization, not for supervision.

## Losses and Metrics

Training loss:

- `mask_bce`: binary cross entropy with logits
- `dice_loss`
- `total_loss = mask_bce + lambda_dice * dice_loss`

Recommended default:

- `lambda_dice = 1.0`

Metrics:

- `mask_iou`
- `dice`
- `foreground_acc`

An optional `center_hit` metric can be derived later, but it is not required for the first delivery.

## Training Script

The training script should follow the same conventions as the other multimodal lessons:

- CLI args for dataset size, image size, mask size, and batch caps
- output directory under `outputs/multimodal/lesson_05_mask_grounding_toy_refexp/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

The sample logger should write:

- query text
- target foreground ratio
- predicted foreground ratio

## Testing Strategy

Focused tests should cover:

- `scripts/run_lesson.py` listing and dry-run resolution for lesson 5
- batch dictionaries containing mask-grounding fields
- model outputs for `mask_logits` and `pred_mask`
- finite mask-grounding loss
- smoke training run writing standard outputs

## Risks and Mitigations

- Risk: foreground regions are too sparse and BCE dominates.
  Mitigation: include dice loss from the start.

- Risk: the lesson feels too similar to generic segmentation.
  Mitigation: keep single-target referring expressions central to the data and model framing.

- Risk: mask supervision becomes too noisy after downsampling.
  Mitigation: use simple object shapes and modest downsampling ratios.

## Success Criteria

Lesson 5 is complete when:

- `lesson_05_mask_grounding_toy_refexp` is discoverable through `scripts/run_lesson.py`
- it runs as a module on CPU
- it contains an independent teaching implementation
- focused tests for discovery, batch contract, model outputs, and smoke training pass
- the lesson clearly demonstrates text-conditioned region grounding
