# Multimodal Lesson 02 BLIP-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_02_blip_compact_captioning` as an independent BLIP-inspired teaching lesson that introduces multimodal fusion, caption generation, and image-text matching without depending on zoo model code.

## Problem

The new multimodal track now has lesson 1 for CLIP-style dual-encoder retrieval, but it does not yet explain the next architectural step: how vision features can be fused with text generation and supervised with an image-text matching objective. The user explicitly chose a dual-task lesson that teaches both captioning and ITM, so lesson 2 should bridge the gap between pure alignment and instruction-style multimodal models.

## Scope

This lesson adds one complete runnable module:

- `tracks/multimodal/lesson_02_blip_compact_captioning/`
- focused tests for lesson discovery and behavior
- track README updates so the new lesson appears in the teaching roadmap

This lesson will implement only two tasks:

- caption generation
- image-text matching

It will not add contrastive pretraining, Q-former style query tokens, or large-scale pretraining abstractions.

## Chosen Architecture

### Recommended approach

The lesson should use a compact BLIP-lite design:

- `VisionEncoder`: convert the synthetic image into a small sequence of visual tokens
- `TextDecoderWithCrossAttention`: generate caption tokens with teacher forcing while attending to visual tokens
- `ITMHead`: classify whether the image and caption match

The image-text matching head should read a fused representation produced by the text side, not an independent dual-encoder branch. This keeps the lesson focused on multimodal fusion rather than reintroducing CLIP.

### Why this approach

This gives lesson 2 a clean educational delta over lesson 1:

- lesson 1 teaches alignment in a shared space
- lesson 2 teaches conditioned generation and multimodal matching

It is also the smallest architecture that still feels recognizably BLIP-like.

## Data Design

The data remains synthetic and CPU-friendly.

Each example contains:

- `image`: a small RGB tensor
- `caption_in_ids`: decoder inputs beginning with BOS
- `caption_out_ids`: shifted targets ending with EOS
- `caption_mask`
- `itm_input_ids`: a caption used for matching classification
- `itm_attention_mask`
- `itm_label`: `1` for matched pairs, `0` for mismatched pairs
- human-readable caption text for sample logging

The image generator should reuse the same structured attribute world as lesson 1:

- color
- shape
- size
- location

The caption should now be sentence-shaped, for example:

- `a small red square at top left`
- `a large blue circle at bottom right`

For ITM negatives, a fixed fraction of examples should replace one or more attributes in the caption while keeping the image unchanged. The negative captions must still be syntactically valid so that the model learns semantic mismatch, not malformed language.

## Model Design

### Visual side

The vision encoder should produce a short sequence of visual tokens rather than one pooled vector. A tiny CNN followed by spatial flattening is enough:

- convolution stack
- small feature map
- flatten spatial positions into tokens
- linear projection to decoder hidden size

### Text side

The text side should be an explicit lesson-local decoder:

- token embedding
- recurrent or Transformer-lite decoder state
- cross-attention from decoder state to visual tokens
- output vocabulary projection

A GRUCell-based decoder with additive cross-attention is recommended. It is easier to read than a full Transformer decoder and keeps the code teaching-first.

### ITM head

The ITM classifier should consume a fused representation from the same decoder path used for captioning. A simple choice is:

- run the decoder over the ITM caption tokens
- mean-pool the decoder hidden states over valid tokens
- classify with a two-layer MLP or linear head

This makes the fusion behavior explicit and avoids adding a separate encoder branch.

## Losses and Metrics

Training loss:

- `caption_loss`: token-level cross entropy with padding ignored
- `itm_loss`: cross entropy over two classes
- `total_loss = caption_loss + itm_loss_weight * itm_loss`

Recommended default:

- `itm_loss_weight = 0.5`

Metrics:

- `caption_token_acc`
- `caption_exact_match`
- `itm_acc`
- `loss`

This provides one metric for text quality, one for full-sequence correctness, and one for multimodal matching.

## Training Script

The lesson should follow existing `tracks/*/lesson_xx/*/train.py` conventions:

- CLI arguments for data size, batch size, widths, epoch count, and smoke batch caps
- `outputs/multimodal/lesson_02_blip_compact_captioning/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

The sample logger should write a few rows containing:

- ground-truth caption
- greedy caption prediction
- ITM ground-truth label
- ITM predicted label

## Testing Strategy

Focused tests should cover:

- `scripts/run_lesson.py` listing and dry-run resolution for lesson 2
- synthetic batch shapes and presence of caption + ITM fields
- forward outputs for caption logits and ITM logits
- finite combined loss
- smoke training run writing standard artifacts

## Risks and Mitigations

- Risk: generation and ITM interfere too much, making smoke training unstable.
  Mitigation: keep vocabulary small, captions templated, and ITM weight modest.

- Risk: negative captions become too trivial.
  Mitigation: generate fluent hard negatives by swapping attributes instead of corrupting syntax.

- Risk: the model becomes too large for lesson readability.
  Mitigation: prefer GRUCell plus additive attention over a full Transformer stack.

## Success Criteria

Lesson 2 is complete when:

- `lesson_02_blip_compact_captioning` is discoverable through `scripts/run_lesson.py`
- the lesson runs as a module on CPU
- the implementation is independent of zoo family files
- focused tests for discovery, batch contract, model outputs, and smoke training pass
- the lesson clearly demonstrates both caption generation and ITM
