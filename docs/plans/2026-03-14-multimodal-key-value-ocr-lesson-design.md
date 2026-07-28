# Multimodal Lesson 12 Key-Value OCR-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm` as an independent teaching lesson for prompt-conditioned key-value OCR on compact document images.

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

What is still missing is a document-style OCR and field extraction lesson. The track explains natural-image VLM patterns well, but it does not yet show how a multimodal model can read structured text from a document image and answer field-specific prompts.

## Candidate Directions

### Option A: Key-value OCR

- synthesize compact document images with rows like `total: 37`
- ask a query like `read total`
- generate the field value as text

Pros:

- clearly document-oriented
- simple enough for CPU
- shows prompt-conditioned OCR and extraction directly

Cons:

- narrower than a full document QA lesson

### Option B: Free-form DocVQA

- synthesize compact receipts or forms
- ask open questions like `what is the total`
- generate the answer

Pros:

- closest to practical document VLM usage

Cons:

- can collapse into generic QA if prompts are too loose

### Option C: Region-based text spotting

- query a spatial text region
- output the text in that region

Pros:

- closer to OCR detection and reading

Cons:

- overlaps more with grounding lessons

## Recommendation

Choose Option A.

It is the clearest teaching step because it adds a new domain, documents, while staying small and prompt-driven. It also keeps the output textual, which connects naturally to lesson 9.

## Task Formulation

Each sample contains:

- one synthetic document image with 3-5 key-value rows
- one query such as `read total`, `read id`, or `read city`

The queried key may be present or absent:

- positive: the document contains the requested field
- negative: the field is missing

The model should output:

- the field value as text tokens when present
- `none` when the field is absent

## Data Design

Each synthetic document should look like a simple card, form, or receipt:

- light background
- several text rows
- each row rendered as `key: value`

Recommended field family:

- `name`
- `id`
- `date`
- `total`
- `city`
- `status`

Each sample should randomly include 3-5 of those fields. The value sets should stay small and closed so the lesson remains CPU-friendly:

- names from a small word list
- cities from a small word list
- status values from a small word list
- id values from a fixed small catalog
- total values from a fixed small catalog
- date values from a fixed small catalog

Each record should include:

- `image`
- `prompt_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `present`
- `query_text`
- `answer_text`

The lesson should render text locally without external OCR dependencies. A tiny built-in bitmap or segment-style font is sufficient.

## Model Design

### Vision side

- tiny CNN over the document image
- flatten to visual tokens
- project into decoder hidden space

### Text side

- token embeddings for the prompt
- decoder-only language model

### Fusion

- prepend visual tokens as a prefix
- decode the field value conditioned on image plus prompt

This keeps the lesson structurally close to lesson 9 while changing the domain and task:

- lesson 9: general prompt-native compact VLM over object images
- lesson 12: document OCR and field extraction as prompt-conditioned text generation

## Losses and Metrics

Training loss:

- standard next-token cross entropy over the answer tokens

Metrics:

- answer token accuracy
- answer exact match
- present accuracy, measured by whether the prediction correctly outputs `none` vs a real value

## Training Script

Follow the same conventions as earlier lessons:

- CLI args for dataset size, image size, and text length
- `outputs/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- query text
- answer ground truth
- answer prediction
- target present
- predicted present

## Success Criteria

Lesson 12 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 12
- the lesson clearly demonstrates prompt-conditioned document OCR and missing-field handling
