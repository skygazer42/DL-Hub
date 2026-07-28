# Multimodal VLM Lessons Design

**Date:** 2026-03-14

**Goal:** Add a new `tracks/multimodal` teaching track to DL-Hub, starting with an independently implemented `lesson_01_clip_compact_retrieval` that teaches the core mechanics of image-text alignment without reusing the zoo model code.

## Problem

The repository now has a local multimodal VLM zoo under `dlhub/multimodal/`, but it does not yet have a lesson-style learning path that teaches how these models work from first principles. The user explicitly wants lesson models to be independent teaching implementations, similar to `tracks/llm/lesson_01_compact_causal_lm_transformer/model.py`, rather than thin wrappers over zoo families such as `clip.py`.

## Scope

This first lesson drop focuses on one complete lesson and one new track shell:

- Create `tracks/multimodal/`
- Create `tracks/multimodal/README.md`
- Create `tracks/multimodal/lesson_01_clip_compact_retrieval/`
- Integrate the new track with `scripts/run_lesson.py` discovery
- Add focused tests for track discovery and lesson behavior

This drop does not yet implement lesson 2 or lesson 3. Those remain part of the track roadmap.

## Chosen Direction

### Recommended lesson sequence

The multimodal teaching track should progress in three stages:

1. `lesson_01_clip_compact_retrieval`
2. `lesson_02_blip_compact_captioning`
3. `lesson_03_llava_compact_instruction_vlm`

The first implementation should only ship lesson 1, because it establishes the shared design vocabulary:

- synthetic multimodal data generation
- independent lesson-local model code
- minimal but meaningful metrics
- a CLI-compatible training loop

### Why start with CLIP-style retrieval

CLIP is the cleanest entry point for multimodal learning because it keeps the task focused on alignment:

- one image encoder
- one text encoder
- a shared embedding space
- a contrastive objective

This avoids the complexity jump of multimodal decoding, teacher forcing, or instruction tuning while still teaching the essential image-text representation story.

## Lesson 1 Design

### Learning objective

The lesson should teach four concepts:

1. image features and text features can be projected into the same embedding space
2. paired examples should have high similarity
3. non-matching pairs in the same batch act as negatives
4. retrieval quality can be measured directly, not only by loss

### Data design

The dataset should be synthetic and deterministic enough to train quickly on CPU.

Each sample contains:

- `image`: a small RGB tensor, for example `3 x 16 x 16`
- `input_ids`: tokenized text describing the same attributes
- `attention_mask`
- `pair_id`: an integer identity for evaluation

Each synthetic concept will combine a small set of visual attributes:

- shape
- color
- size
- position or border pattern

The image renderer does not need external dependencies beyond PyTorch. It can draw simple rectangles, crosses, or filled regions directly into tensors. The paired caption can follow a stable template such as `"red square small top-left"`.

This keeps the modality bridge obvious and makes retrieval metrics interpretable.

### Model design

The lesson-local `model.py` should not import or wrap `dlhub.multimodal.vlm.clip`. It should define its own teaching-sized modules:

- `VisionEncoder`: a tiny CNN that maps `3 x H x W` to a feature vector
- `TextEncoder`: token embedding + mean pooling over valid tokens
- `ProjectionHead`: linear projection to a shared embedding space
- `CompactCLIPModel`: returns normalized image and text embeddings plus similarity logits

The forward API should stay simple and lesson-friendly:

- input: a batch dictionary with image and token tensors
- output: a dictionary with `image_embed`, `text_embed`, `logits_per_image`, `logits_per_text`

### Loss and metrics

Training uses the standard symmetric contrastive loss:

- cross entropy over `logits_per_image`
- cross entropy over `logits_per_text`
- average the two

Evaluation should report:

- `loss`
- `image_to_text_acc`
- `text_to_image_acc`

For this compact lesson, top-1 batch retrieval accuracy is enough.

### Training script

The training script should follow existing lesson conventions:

- parse CLI arguments
- build dataloaders
- save config and metrics
- train for a few epochs
- write a checkpoint

Unlike the zoo CLI, the lesson should optimize a real compact model end-to-end. The script should also record a small `samples.jsonl` file with a few captions and predicted retrieval indices so the run artifact is inspectable.

## Track Layout

The new track should follow the same pattern as other tracks:

- `tracks/multimodal/__init__.py`
- `tracks/multimodal/README.md`
- `tracks/multimodal/lesson_01_clip_compact_retrieval/__init__.py`
- `tracks/multimodal/lesson_01_clip_compact_retrieval/data.py`
- `tracks/multimodal/lesson_01_clip_compact_retrieval/model.py`
- `tracks/multimodal/lesson_01_clip_compact_retrieval/train.py`
- `tracks/multimodal/lesson_01_clip_compact_retrieval/README.md`

This makes the lesson discoverable via:

`python scripts/run_lesson.py multimodal --list`

## Testing Strategy

The first delivery needs focused tests only:

- `tests/test_scripts_run_lesson.py`
  - `multimodal` appears in track listing
  - `lesson_01_clip_compact_retrieval` appears in lesson listing
  - dry-run resolves `tracks.multimodal.lesson_01_clip_compact_retrieval.train`
- `tests/test_tracks_multimodal_clip.py`
  - synthetic batch shapes are stable
  - the teaching model returns the expected keys and tensor shapes
  - the contrastive loss is finite
  - a minimal training smoke run writes standard outputs

## Risks and Mitigations

- Risk: synthetic data is too random, so retrieval accuracy does not improve.
  Mitigation: use a structured attribute grammar with low ambiguity and repeatable templates.

- Risk: the model accidentally mirrors zoo abstractions and becomes hard to teach.
  Mitigation: keep lesson-local classes explicit and small, with names tied to educational concepts.

- Risk: training smoke is too slow for tests.
  Mitigation: default test config to very small image size, short captions, and a capped number of train and eval batches.

## Success Criteria

This first lesson drop is complete when:

- `tracks/multimodal` is discoverable by `scripts/run_lesson.py`
- `lesson_01_clip_compact_retrieval` runs as a module
- the lesson contains independent `data.py`, `model.py`, and `train.py`
- focused tests for the new track and lesson pass
- the lesson can complete a CPU smoke training run and write normal run artifacts
