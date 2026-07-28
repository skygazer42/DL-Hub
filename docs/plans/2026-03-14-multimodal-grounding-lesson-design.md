# Multimodal Lesson 04 Grounding-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_04_grounding_compact_refexp` as an independent teaching lesson for single-target referring expression grounding with grid-cell localization plus box decoding.

## Problem

The multimodal track now covers:

- lesson 1: CLIP-style alignment
- lesson 2: BLIP-lite captioning plus ITM
- lesson 3: LLaVA-lite single-turn visual instruction answering

What is still missing is a region-aware multimodal lesson that shows how language can guide spatial localization in an image. The user chose bbox grounding, and the most teaching-friendly form is a grid-based grounding lesson that decomposes the problem into classification plus local box regression.

## Scope

This lesson adds one full runnable module:

- `tracks/multimodal/lesson_04_grounding_compact_refexp/`
- focused tests for discovery, batch contract, model outputs, and smoke training
- an update to `tracks/multimodal/README.md` so lesson 4 appears in the progression

This lesson will support:

- single-target grounding
- referring expressions over synthetic objects
- grid-cell localization plus box-offset decoding

It will not support:

- phrase-to-region ranking over candidates
- multi-object grounding
- segmentation masks
- region captioning

## Chosen Direction

### Recommended formulation

Use a single-target grounding task with grid-based localization:

- input: `image + referring expression`
- output: target grid cell plus local box offsets

This is more stable and more interpretable for a teaching lesson than direct bbox regression.

### Why not direct box regression

Direct regression is concise but hides the decomposition that is useful to teach:

- spatial search over candidate locations
- local coordinate refinement around the chosen location

The grid-based form makes the conditioning effect of the text easier to understand.

## Data Design

Each synthetic image should contain multiple objects, for example 2 to 4, selected from the familiar attribute world:

- color
- shape
- size
- location

One object is chosen as the target. The referring expression should uniquely describe it, with templates such as:

- `find the red square`
- `locate the small blue circle`
- `point to the large yellow cross`
- `find the green object at top left`

Each record contains:

- `image`
- `input_ids`
- `attention_mask`
- `target_cell`
- `target_box`
- `target_delta`
- `query_text`

The target annotations should include:

- normalized bbox
- target grid index
- local offsets inside the target grid cell

## Model Design

### Vision side

The vision encoder should preserve spatial structure:

- tiny CNN backbone
- spatial feature map rather than pooled embedding

### Text side

The text encoder can stay lightweight:

- token embedding
- masked mean pooling

This is enough because the linguistic complexity of referring expressions is intentionally small.

### Fusion

The text representation should be broadcast across the spatial grid and fused with each visual location. This keeps the lesson explicit: the text changes the interpretation of every spatial cell.

### Grounding head

The output head should produce:

- `cell_logits`: one logit per spatial cell
- `box_deltas`: per-cell local box predictions

The training loop will select the target cell and supervise its box prediction. Decoding should convert the chosen cell plus offsets back into a normalized bbox.

## Losses and Metrics

Training loss:

- `cell_loss`: cross entropy over the target cell
- `box_loss`: smooth L1 over the target cell box prediction
- `total_loss = cell_loss + lambda_box * box_loss`

Recommended default:

- `lambda_box = 2.0`

Metrics:

- `cell_acc`
- `bbox_l1`
- `center_acc`

`center_acc` should check whether the predicted box center falls inside the GT box. This is a simple, interpretable compact grounding metric.

## Training Script

The lesson should follow the same conventions as the existing multimodal lessons:

- CLI args for dataset size, image size, grid size, and batch caps
- output directory under `outputs/multimodal/lesson_04_grounding_compact_refexp/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

The sample logger should write:

- query text
- GT bbox
- predicted bbox
- GT cell
- predicted cell

## Testing Strategy

Focused tests should cover:

- `scripts/run_lesson.py` listing and dry-run resolution for lesson 4
- batch dictionaries containing grounding fields
- model outputs for cell logits, box deltas, and decoded boxes
- finite grounding loss
- smoke training run writing standard outputs

## Risks and Mitigations

- Risk: referring expressions are ambiguous.
  Mitigation: generate scenes and queries so the target is unique by construction.

- Risk: direct bbox decoding becomes hard to reason about.
  Mitigation: keep grid-cell classification explicit and decode only from the selected cell.

- Risk: the lesson drifts too close to detection rather than grounding.
  Mitigation: always condition localization on text and use single-target referring expressions.

## Success Criteria

Lesson 4 is complete when:

- `lesson_04_grounding_compact_refexp` is discoverable through `scripts/run_lesson.py`
- it runs as a module on CPU
- it contains an independent teaching implementation
- focused tests for discovery, batch contract, model outputs, and smoke training pass
- the lesson clearly demonstrates text-conditioned spatial grounding
